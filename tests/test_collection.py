import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from r2ai_autoresearch.collection.pool import GeneratedSourcePool
from r2ai_autoresearch.research.config import validate

class CollectionTests(unittest.TestCase):
    def test_full_seed_coverage_retry_only_failures_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);parent=root/'parent';parent.write_text('checkpoint')
            rows=[{'id':str(i),'source':'original','kind':'harmful','split':'train','prompt':'seed '+str(i)} for i in range(3)]
            c={'seed':42,'sources':['attack'],'collection':{'batch_size':2,'max_candidates_per_source':6,'providers':{'attack':{'kind':'pool'}}}}
            calls=[]
            def worker(name,op,parent,items):
                calls.extend(r['seed_id'] for r in items)
                return {'outputs':[{'id':r['id'],'text':'safe' if r['seed_id']=='2' else 'unsafe','hit_token_limit':False} for r in items]}
            loop=SimpleNamespace(root=root/'run',c=c,data={'rows':rows},client=None,worker=worker)
            def gen(loop,source,cfg,seed,attempt,directory):
                return {'id':seed['id']+str(attempt),'seed_id':seed['id'],'prompt':seed['prompt']+str(attempt),'kind':'harmful','split':'train','source':source}
            cfg={'attack':{'kind':'darwin_p2','seed_source':'original','attempts_per_seed':2}}
            with patch('r2ai_autoresearch.collection.pool.generate',side_effect=gen),patch('r2ai_autoresearch.collection.pool.judge',side_effect=lambda c,r,o:[{'unsafe':o[0]=='unsafe'}]):
                p=GeneratedSourcePool(loop,'round-1',parent,cfg)
                self.assertEqual(len(p.ensure('attack',1,'pilot')),1)
                self.assertEqual(calls.count('2'),2);self.assertEqual(calls.count('0'),1);self.assertEqual(calls.count('1'),1)
                self.assertEqual(len(p.available('attack')),2)
                before=len(calls);q=GeneratedSourcePool(loop,'round-1',parent,cfg);q.ensure('attack',9,'resume');self.assertEqual(before,len(calls))
                parent.write_text('changed')
                with self.assertRaisesRegex(ValueError,'stale|Stale'):GeneratedSourcePool(loop,'round-1',parent,cfg)

    def test_invalid_generator_config_is_rejected(self):
        root=Path(__file__).resolve().parents[1];c=json.loads((root/'config.json').read_text())
        c['data']['providers']={'unknown':{}}
        with self.assertRaises(ValueError):validate(c)

if __name__=='__main__':unittest.main()
