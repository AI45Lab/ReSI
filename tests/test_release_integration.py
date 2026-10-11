import copy
import io
import urllib.error
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import test_research_data_eval as data_tests
from r2ai_autoresearch.research.data import prepare, split_safety_pool
from r2ai_autoresearch.research.config import validate
from r2ai_autoresearch.research.backend import Backend
from r2ai_autoresearch.research.responses import split_response, decode_response
from r2ai_autoresearch.research.reward import judge_rewards
from r2ai_autoresearch.portable.source_pool import SourcePool

ROOT=Path(__file__).resolve().parents[1]

class ReleaseIntegrationTests(unittest.TestCase):
    def test_replay_never_enters_current_source_verification_or_benign_pool(self):
        rows=[{'id':'current','source':'a','split':'train','kind':'harmful','prompt':'current prompt'},
              {'id':'old-h','source':'a','split':'train','kind':'harmful','prompt':'old harmful','mixture_role':'replay'},
              {'id':'old-b','source':'a','split':'train','kind':'benign','prompt':'old benign','mixture_role':'replay'}]
        with tempfile.TemporaryDirectory() as tmp:
            b=object.__new__(Backend);b.root=Path(tmp);b.data={'rows':rows};b.original=[]
            b.c={'sources':['a'],'seed':42,'collection':{}}
            with patch('r2ai_autoresearch.portable.source_pool.sha256_path',return_value='fixture'):
                b.pool=SourcePool(b,'round',Path(tmp))
            self.assertEqual([r['id'] for r in b.pool.seeds('a')],['current'])
            b.pool.states['a']['accepted']=[rows[0]]
            current=b.current_rows({'alpha':{'a':1},'benign_repeats':1})
            self.assertEqual([r['id'] for r in current],['current'])
            mixed=b.train_rows({'alpha':{'a':1},'benign_repeats':1,'paired':False,
                               'mixture':{'current':.5,'replay':.5,'general':0.}})
            self.assertEqual(sum(r.get('mixture_role')=='replay' for r in mixed),1)

    def test_provider_content_filter_is_not_retried_or_scored(self):
        from r2ai_autoresearch.portable.service import Client, ServicePolicyError
        body=io.BytesIO(json.dumps({'error':{'code':'content_filter','message':'test-secret private-content'}}).encode())
        error=urllib.error.HTTPError('http://example.test',400,'blocked',{},body)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(Client,'endpoint',return_value=('http://example.test/v1','judge','test-secret')), \
             patch('r2ai_autoresearch.portable.service.urllib.request.urlopen',side_effect=error) as request:
            with self.assertRaisesRegex(ServicePolicyError,'no score was recorded'):
                Client(folder,5).call('JUDGE','fixture',{})
            self.assertEqual(request.call_count,1)
            paths=list(Path(folder).glob('*.json'));self.assertEqual(len(paths),1)
            receipt=json.loads(paths[0].read_text())
            self.assertEqual(receipt['provider_code'],'content_filter')
            self.assertEqual(receipt['status'],'failed')
            self.assertNotIn('test-secret',paths[0].read_text())
            self.assertNotIn('private-content',paths[0].read_text())

    def test_transient_api_retry_counts_each_request_and_reuses_success(self):
        from r2ai_autoresearch.portable.service import Client
        response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'{"ok":true}'}}]}).encode())
        with tempfile.TemporaryDirectory() as folder, patch.object(Client,'endpoint',return_value=('http://example.test/v1','judge','test-secret')), \
             patch('r2ai_autoresearch.portable.service.time.sleep'), \
             patch('r2ai_autoresearch.portable.service.urllib.request.urlopen',side_effect=[urllib.error.HTTPError('http://example.test',503,'unavailable',{},None),io.BytesIO(b'{"error":"upstream unavailable"}'),response]) as request:
            client=Client(folder,3)
            self.assertEqual(client.call('JUDGE','fixture',{}),{'ok':True})
            self.assertEqual(client.call('JUDGE','fixture',{}),{'ok':True})
            self.assertEqual(request.call_count,3)
            receipts=[json.loads(p.read_text()) for p in sorted(Path(folder).glob('call-*.json'))]
            self.assertEqual([r['status'] for r in receipts],['failed','failed','completed'])
            self.assertNotIn('test-secret',''.join(p.read_text() for p in Path(folder).glob('*.json')))

    def test_api_retries_do_not_bypass_budget_or_retry_permission_errors(self):
        from r2ai_autoresearch.portable.service import Client,ServiceError
        for status,budget,expected in [(403,5,1),(503,2,2),(503,10,3)]:
            with self.subTest(status=status,budget=budget), tempfile.TemporaryDirectory() as folder, \
                 patch.object(Client,'endpoint',return_value=('http://example.test/v1','judge','')), \
                 patch('r2ai_autoresearch.portable.service.time.sleep'), \
                 patch('r2ai_autoresearch.portable.service.urllib.request.urlopen',side_effect=urllib.error.HTTPError('http://example.test',status,'failure',{},None)) as request:
                with self.assertRaises(ServiceError):Client(folder,budget).call('JUDGE','fixture',{})
                self.assertEqual(request.call_count,expected)
                self.assertEqual(len(list(Path(folder).glob('call-*.json'))),expected)

    def test_shared_base_survives_source_overlap_without_duplicate_prompt(self):
        fixture=data_tests.DataEvalTests();fixture.setUp()
        _,remaining=split_safety_pool(fixture.safety,1,42)
        row=remaining[0]
        train={'rows':[{**row,'id':'attack-overlap','seed_id':row['id'],'source':'darwin','split':'train'}]}
        result=prepare(train,fixture.safety,fixture.panels,{},fixture.counts,original_source='original')
        originals=[r for r in result['rows'] if r.get('source')=='original' or 'original' in r.get('shared_base_sources',[])]
        self.assertEqual(len(originals),len(remaining))
        self.assertEqual(len({r['prompt'] for r in result['rows']}),len(result['rows']))
        self.assertEqual(next(r for r in originals if r['id']=='attack-overlap')['source'],'darwin')
        with tempfile.TemporaryDirectory() as d:
            class Loop:pass
            b=Loop();b.root=Path(d);b.data=result
            b.c={'sources':['darwin','original'],'seed':42,'collection':{}}
            with patch('r2ai_autoresearch.portable.source_pool.sha256_path',return_value='fixture'):
                pool=SourcePool(b,'round-1',Path(d))
            self.assertEqual(len(pool.seeds('original')),len(remaining))
            self.assertEqual(len(pool.seeds('darwin')),1)

    def test_public_limits_match_backend_initialization(self):
        cfg=json.loads((ROOT/'config.smoke.json').read_text())
        data=json.loads((ROOT/cfg['data']['path']).read_text())
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model';model.mkdir();(model/'config.json').write_text('{"model_type":"qwen3"}')
            for section,key,value in [('training','max_new_tokens',8192),('runtime','max_api_calls',100001)]:
                c=copy.deepcopy(cfg);c[section][key]=value;validate(c)
                Backend(c,data,model,Path(d)/key)
        for value in (0,True,1.2):
            c=copy.deepcopy(cfg);c['runtime']['max_api_calls']=value
            with self.assertRaises(ValueError):validate(c)

    def test_response_formats_and_unfinished_reasoning(self):
        for raw in ('<think>private</think>final','[THINK]private[/THINK]final','private</think>final','private[/THINK]final'):
            self.assertEqual(split_response(raw)['final_answer'],'final')
        for raw in ('<think>private','[THINK]private'):
            self.assertFalse(split_response(raw)['final_complete'])
            self.assertEqual(split_response(raw)['final_answer'],'')
        self.assertEqual(split_response('answer')['final_answer'],'answer')
        self.assertFalse(split_response('private',reasoning_started=True)['final_complete'])

    def test_decode_keeps_boundaries_until_after_channel_split(self):
        class Tokenizer:
            all_special_tokens=['[THINK]','[/THINK]','</s>']
            def decode(self,ids,skip_special_tokens):
                return 'private final' if skip_special_tokens else '[THINK]private[/THINK]final</s>'
        out=decode_response(Tokenizer(),[1,2,3])
        self.assertEqual(out['text'],'final');self.assertEqual(out['reasoning'],'private')
        self.assertIn('[/THINK]',out['raw_text'])

    def test_reward_judges_final_only(self):
        received=[]
        def fake(client,batch,group,rubric):
            received.append(batch[0][1]['text']);i=batch[0][0]['id']
            return {i:{'id':i,'score':1,'reason':'fixture'}}
        with patch('r2ai_autoresearch.research.reward.judge_batch',side_effect=fake):
            values,_=judge_rewards(None,[{'id':'a','prompt':'request','kind':'harmful'}],['[THINK]private[/THINK]final'])
        self.assertEqual(values,[1]);self.assertEqual(received,['final'])

if __name__=='__main__':unittest.main()
