import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from r2ai_autoresearch.research.a3_workflow import DEFAULTS,build,split_hypotheses,hypothesis_weights
from r2ai_autoresearch.research.config import load

class SafinA3Tests(unittest.TestCase):
    def test_split_is_by_hypothesis_and_reproducible(self):
        h=[{'id':f'h{i}'} for i in range(60)]
        a=split_hypotheses(h,.7,.2,42)
        self.assertEqual(a,split_hypotheses(h,.7,.2,42))
        self.assertEqual([sum(x['split']==s for x in a) for s in ['train','validation','ood']],[42,12,6])
        with self.assertRaises(ValueError):split_hypotheses(h[:2],.7,.2,42)

    def test_historical_weight_boundaries(self):
        self.assertEqual(hypothesis_weights(0,DEFAULTS['weight_bands']),{'harmful':2.,'benign':2.5})
        self.assertEqual(hypothesis_weights(.075,DEFAULTS['weight_bands'])['harmful'],4.5)
        self.assertEqual(hypothesis_weights(.08,DEFAULTS['weight_bands'])['harmful'],6.5)
        self.assertEqual(hypothesis_weights(1,DEFAULTS['weight_bands'])['harmful'],10.)

    def test_build_keeps_validation_and_ood_out_of_targets_and_weighting(self):
        class Client:
            def __init__(self):self.target_prompts=[]
            def validated(self,role,instruction,payload,validator,**kwargs):
                if 'observations' in payload:
                    v={'behaviors':[{'id':'b0','title':'boundary','description':'boundary','safety_risk_pattern':'boundary','basis_ids':['seed']}]}
                elif 'required_ids' in payload:
                    v={'hypotheses':[dict(id=i,behavior_id='b0',title=i,hypothesis='hypothesis '+i,testable_prediction='prediction',pairing_principle='contrast') for i in payload['required_ids']]}
                elif 'hypothesis' in payload:
                    h=payload['hypothesis']['id'];v={'pairs':[dict(harmful_prompt='harmful '+h,benign_prompt='benign '+h,pair_rationale='contrast')]}
                elif 'rows' in payload:
                    v={'items':[{'id':r['id'],'predicted_kind':r['kind']} for r in payload['rows']]}
                else:
                    self.target_prompts.append(payload['prompt']);v={'response':'safe supervision'}
                return validator(v)
        with tempfile.TemporaryDirectory() as d:
            cfg=copy.deepcopy(DEFAULTS);cfg.update(behaviors=1,growth_rounds=1,hypotheses_per_round=3,pairs_per_hypothesis=1,train_fraction=.34,validation_fraction=.34)
            seed={'id':'seed','prompt':'seed prompt','kind':'harmful','split':'train','source':'direct'}
            calls=[]
            def worker(name,operation,parent,rows):
                calls.append(rows)
                return {'outputs':[{'text':'safe response','hit_token_limit':False} for r in rows]}
            loop=SimpleNamespace(root=Path(d),c={'a3':{'safin':cfg},'seed':42},data={'rows':[seed]},worker=worker,client=Client())
            with patch('r2ai_autoresearch.research.a3_workflow.judge',side_effect=lambda c,rows,texts:[{'unsafe':False,'refused':r['kind']=='harmful'} for r in rows]):
                result=build(loop,'direct','base',[seed])
                self.assertEqual(build(loop,'direct','base',[seed]),result)
            self.assertEqual(len(calls),2)  # Seed observation and training-hypothesis weighting only; cache reuses both.
            self.assertEqual(len(result['train']),2)
            self.assertEqual({r['split'] for r in calls[1]},{'train','validation','ood'})
            self.assertEqual({x['row']['split'] for x in result['probe_evidence']},{'validation','ood'})
            self.assertFalse(set(loop.client.target_prompts)&{x['row']['prompt'] for x in result['probe_evidence']})
            self.assertEqual(len(result['weights']),1)

    def test_release_and_smoke_use_same_workflow_at_different_sizes(self):
        root=Path(__file__).resolve().parents[1]
        for path in ['config.json','config.smoke.json']:
            c=load(root/path);self.assertEqual(c['a3']['workflow'],'safin')
