import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from r2ai_autoresearch.research.backend import Backend
from r2ai_autoresearch.research.controller import Controller
from r2ai_autoresearch.research.reward import reward_from_judgments
from r2ai_autoresearch.research.config import load

ROOT=Path(__file__).resolve().parents[1]
BASE={'primary_asr':.5,'secondary_asr':.5,'benign_response':1.,'instruction_constraint':1.}

class Client:
    def validated(self,role,instruction,payload,validator,**kwargs):
        return validator(self.call(role,instruction,payload,**kwargs))
    def call(self,role,instruction,payload,**kwargs):
        return {'items':[{'choice':x['id'],'hypothesis':'fixture','expected':'fixture','falsifier':'fixture'} for x in payload['candidates']]}

class Pool:
    def __init__(self,loop,cycle,parent):self.loop=loop;self.parent=str(parent)
    def available(self,source):return [r for r in self.loop.data['rows'] if r['split']=='train' and r['kind']=='harmful' and r.get('source')==source]
    def ensure(self,source,count,reason):return self.available(source)[:count]
    def seeds(self,source):return self.available(source)

class Bridge(Backend):
    def event(self,*args,**kwargs):pass
    def evaluate(self,name,parent):return copy.deepcopy(BASE if name=='initial' else {**BASE,'primary_asr':.2})
    def worker(self,name,operation,parent,rows,candidate=None):
        self.calls.append({'name':name,'operation':operation,'parent':str(parent),'rows':rows})
        if operation=='grpo_screen':return {'selected_ids':[r['id'] for r in rows],'screened_prompts':len(rows)}
        return {'checkpoint':name}

class BackendTests(unittest.TestCase):
    def test_proposal_uses_the_gate_metric_and_feasible_gain_bound(self):
        class CapturingClient(Client):
            def call(self,role,instruction,payload,**kwargs):
                self.instruction=instruction;self.payload=payload
                return super().call(role,instruction,payload,**kwargs)
        b=object.__new__(Backend);b.config=load(ROOT/'config.smoke.json');b.client=CapturingClient()
        for primary,secondary,metric,bound in [(0.,.4,'secondary_asr',.4),(.05,.4,'primary_asr',.05),(0.,0.,'secondary_asr',0.)]:
            baseline={**BASE,'primary_asr':primary,'secondary_asr':secondary}
            b.propose('trial',{'method':'critic_sft'},baseline)
            objective=b.client.payload['safety_objective']
            self.assertEqual(objective['metric'],metric)
            self.assertEqual(objective['baseline_asr'],bound)
            self.assertEqual(objective['maximum_possible_gain'],bound)
            self.assertEqual(b.client.payload['parent_metrics'],baseline)
            self.assertIn('never predict an ASR below zero',b.client.instruction)

    def test_missing_proposal_is_repaired_without_changing_candidates(self):
        from r2ai_autoresearch.portable.service import Client as RealClient
        with tempfile.TemporaryDirectory() as tmp:
            b=object.__new__(Backend);b.config=load(ROOT/'config.smoke.json');b.client=RealClient(tmp,10)
            good={'items':[dict(choice='trial',hypothesis='h',expected='e',falsifier='f')]}
            with patch.object(b.client,'call',side_effect=[{'items':[]},good]) as calls:
                result=b.propose('trial',{'method':'critic_sft'},BASE)
            self.assertEqual(result['choice'],'trial')
            self.assertEqual(calls.call_count,2)
            self.assertIn('trial',calls.call_args.args[2]['format_correction'])

    def test_collection_limit_cannot_silently_truncate_full_source_pool(self):
        c=load(ROOT/'config.smoke.json')
        data=json.loads((ROOT/'data/demo/tasks.json').read_text())
        with tempfile.TemporaryDirectory() as tmp,patch('r2ai_autoresearch.research.backend.SourcePool',Pool):
            model=Path(tmp)/'model';model.mkdir();(model/'config.json').write_text('{"model_type":"qwen3"}')
            b=Bridge(c,data,model,Path(tmp)/'run')
            b.c['collection']['max_candidates_per_source']=1
            with self.assertRaisesRegex(ValueError,'complete declared pool'):
                b.begin_round(0,model,False)

    def test_real_bridge_routes_selected_method_and_s4(self):
        c=load(ROOT/'config.smoke.json');c['search'].update(methods=['critic_sft'],max_scaling_probes=0)
        data=json.loads((ROOT/'data/demo/tasks.json').read_text())
        with tempfile.TemporaryDirectory() as tmp,patch('r2ai_autoresearch.research.backend.SourcePool',Pool),patch('r2ai_autoresearch.research.supervision.build',side_effect=lambda loop,rows,parent,name:[{**r,'target':'fixture'} for r in rows]):
            (Path(tmp)/'model').mkdir();(Path(tmp)/'model/config.json').write_text('{"model_type":"qwen3"}')
            b=Bridge(c,data,Path(tmp)/'model',Path(tmp)/'run');b.client=Client();b.calls=[]
            result=Controller(c,b).run()
            self.assertEqual(len(result['history'][0]['full_trials']),4)
            sft=[r for r in b.calls if r['operation']=='sft'];grpo=[r for r in b.calls if r['operation']=='grpo']
            self.assertEqual(len(sft),4);self.assertEqual(len(grpo),2)
            self.assertTrue(all('/S4/' in r['name'] for r in grpo))
            self.assertTrue(all(all(row['kind']!='utility' for row in r['rows']) for r in b.calls))

    def test_binary_reward_distinguishes_score4_from_benchmark_asr(self):
        self.assertEqual(reward_from_judgments('harmful',3),1)
        self.assertEqual(reward_from_judgments('harmful',4),0)
        self.assertEqual(reward_from_judgments('benign',1,'3_partial_refusal'),0)
        self.assertEqual(reward_from_judgments('benign',1,'1_full_compliance'),1)
        with self.assertRaises(ValueError):reward_from_judgments('utility',1)

    def test_relative_data_paths_follow_config_not_cwd(self):
        c=load(ROOT/'config.smoke.json')
        self.assertEqual(Path(c['data']['path']),ROOT/'data/demo/tasks.json')

    def test_s2_sampling_and_duplicate_rejection(self):
        c=load(ROOT/'config.smoke.json')
        data=json.loads((ROOT/'data/demo/tasks.json').read_text())
        rows=[r for r in data['rows'] if r['split']=='train' and r['kind']=='harmful'][:2]
        class PairClient:
            def validated(self,*args,**kwargs):return {'prompt':'A new legitimate request for this test.'}
            def call(self,*args,**kwargs):return {'kind':'benign'}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'model').mkdir();(Path(tmp)/'model/config.json').write_text('{"model_type":"qwen3"}')
            b=Bridge(c,data,Path(tmp)/'model',Path(tmp)/'run');b.client=PairClient();b.pool=Pool(b,'round-1',b.model)
            result=b.paired_rows(rows[:1],'pair-test',repeats=3)
            self.assertEqual(len(result),4)
            self.assertEqual(len([r for r in result if r['kind']=='benign']),3)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                b.paired_rows(rows,'duplicate-test',repeats=2)

    def test_bad_generated_training_data_is_not_scored_as_safe(self):
        from r2ai_autoresearch.portable.errors import CandidateDataError
        c=load(ROOT/'config.smoke.json');data=json.loads((ROOT/'data/demo/tasks.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'model').mkdir();(Path(tmp)/'model/config.json').write_text('{"model_type":"qwen3"}')
            b=Bridge(c,data,Path(tmp)/'model',Path(tmp)/'run')
            with patch.object(b,'_train',side_effect=CandidateDataError('repeated generated prompts')):
                result=b.train('trial',{},b.model)
            self.assertIsNone(result['checkpoint']);self.assertEqual(result['issue'],'candidate_data')
            with patch.object(b,'_train',side_effect=RuntimeError('GPU offline')):
                with self.assertRaises(RuntimeError):b.train('other',{},b.model)

    def test_s3_preserves_safety_replay_ratio(self):
        b=object.__new__(Backend);b.config=load(ROOT/'config.json')
        b.data={'rows':[{'split':'train','kind':'utility','target':'reference'}]}
        before={'method':'critic_sft','mixture':{'current':.75,'replay':.25,'general':0.},
                'learning_rate':2e-5,'steps':128}
        options=b.refinements({'recipe':before,'gate':{'issues':['instruction_following']}})
        mix=next(o['recipe']['mixture'] for o in options if o['id']=='S3-general-0.2')
        self.assertAlmostEqual(mix['current'],.6)
        self.assertAlmostEqual(mix['replay'],.2)
        self.assertAlmostEqual(mix['general'],.2)

    def test_s2_pairs_before_mixture_and_reuses_verified_prompts(self):
        class PairClient:
            def __init__(self):self.calls=0
            def validated(self,role,instruction,payload,check):
                self.calls+=1;return check({'prompt':'Benign counterpart '+payload['prompt']})
            def call(self,*args,**kwargs):return {'kind':'benign'}
        rows=[{'id':str(i),'source':'a','split':'train','kind':'harmful','prompt':'request '+str(i),'target':'harmful target'} for i in range(2)]
        rows += [{'id':'b','source':'a','split':'train','kind':'benign','prompt':'benign'},
                 {'id':'r','split':'train','kind':'benign','prompt':'replay','mixture_role':'replay'},
                 {'id':'u','split':'train','kind':'utility','prompt':'general','target':'reference'}]
        with tempfile.TemporaryDirectory() as tmp:
            b=object.__new__(Backend);b.data={'rows':rows};b.original=[];b.pool=Pool(b,'round',tmp)
            b.c={'seed':42};b.root=Path(tmp);b.client=PairClient();b.exclusion_rows=lambda:[]
            recipe={'alpha':{'a':2},'paired':True,'benign_repeats':2,
                    'mixture':{'current':.5,'replay':.25,'general':.25}}
            result=b.train_rows(recipe,'trial1')
            self.assertEqual(len(result),16)
            self.assertEqual(sum(r['kind']=='utility' for r in result),4)
            self.assertEqual(sum(r.get('mixture_role')=='replay' for r in result),4)
            self.assertTrue(all('target' not in r for r in result if r.get('paired_id')))
            again=b.train_rows(recipe,'trial2')
            self.assertEqual(result,again);self.assertEqual(b.client.calls,2)
