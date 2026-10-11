"""Bounded selection tests; fake scores are policy tests, not model evidence."""
import copy
from pathlib import Path
import unittest
from r2ai_autoresearch.research.controller import Controller
from r2ai_autoresearch.research.config import load
from r2ai_autoresearch.portable.gpu import checkpoint_steps
from r2ai_autoresearch.portable.errors import CandidateEvaluationError
from r2ai_autoresearch.research.backend import Backend

BASE={'primary_asr':.5,'secondary_asr':.5,'benign_response':1.,'instruction_constraint':1.}
ROOT=Path(__file__).resolve().parents[1]
class Fake:
    def __init__(self,metrics):self.metrics=metrics;self.names=[];self.records=[]
    def propose(self,*args):return {}
    def train(self,name,recipe,parent):
        self.recipe=recipe
        if '_checkpoint_selection' not in recipe:return {'checkpoint':'final'}
        return {'checkpoint':'final','checkpoint_candidates':[{'checkpoint':x,'step':i+1} for i,x in enumerate(self.metrics)]}
    def evaluate(self,name,checkpoint):
        self.names.append(name)
        v=self.metrics[checkpoint]
        if isinstance(v,Exception):raise v
        return copy.deepcopy(v)
    def record(self,name,record):self.records.append(record)

class CheckpointTests(unittest.TestCase):
    def config(self):
        c=load(ROOT/'config.smoke.json');c['training']['checkpoint_selection']={'enabled':True,'max_checkpoints':3}
        return c
    def trial(self,metrics,phase='critic_sft'):
        b=Fake(metrics);c=Controller(self.config(),b)
        result=c.trial('branch/trial-0',{'method':'critic_sft'},'parent',BASE,BASE,False,phase)
        return result,c,b
    def test_schedule_is_bounded_unique_and_includes_final(self):
        self.assertEqual(checkpoint_steps(10,3),[4,7,10])
        self.assertEqual(checkpoint_steps(2,3),[1,2])
        self.assertEqual(checkpoint_steps(10,1),[10])
    def test_earlier_eligible_survives_better_safety_with_utility_failure(self):
        r,c,b=self.trial({'early':{**BASE,'primary_asr':.3},'final':{**BASE,'primary_asr':.1,'instruction_constraint':.8}})
        self.assertEqual(r['checkpoint'],'early');self.assertTrue(r['gate']['eligible'])
        self.assertEqual(len(c.records),1);self.assertEqual(len(b.names),2)
        self.assertNotIn('_checkpoint_selection',r['recipe'])
    def test_best_eligible_wins_not_first(self):
        r,c,b=self.trial({'early':{**BASE,'primary_asr':.3},'final':{**BASE,'primary_asr':.2}})
        self.assertEqual(r['checkpoint'],'final')
    def test_invalid_evaluation_does_not_hide_earlier_eligible(self):
        r,c,b=self.trial({'early':{**BASE,'primary_asr':.3},'final':CandidateEvaluationError('truncated')})
        self.assertEqual(r['checkpoint'],'early')
        self.assertEqual(r['checkpoint_evaluations'][1]['status'],'evaluation_incomplete')
    def test_no_eligible_keeps_failure_for_refinement(self):
        r,c,b=self.trial({'early':{**BASE,'primary_asr':.3,'instruction_constraint':.8},'final':{**BASE,'primary_asr':.2,'instruction_constraint':.8}})
        self.assertFalse(r['gate']['eligible']);self.assertEqual(r['checkpoint'],'final')
    def test_pilot_stays_final_only(self):
        r,c,b=self.trial({'final':{**BASE,'primary_asr':.3}},phase='pilot')
        self.assertNotIn('_checkpoint_selection',b.recipe);self.assertEqual(len(b.names),1)
    def test_s4_starts_from_selected_earlier_sft_and_keeps_trial_budget(self):
        from test_research_protocol import Fake as ProtocolFake
        c=self.config();c['search'].update(max_rounds=1,shortlist_size=1,max_scaling_probes=0,methods=['critic_sft'])
        class Checkpoints(ProtocolFake):
            def train(self,name,recipe,parent):
                self.calls.append((name,copy.deepcopy(recipe),parent))
                if '/full/' in name:
                    return {'checkpoint':name+'/last','checkpoint_candidates':[
                        {'checkpoint':name+'/early','step':1},{'checkpoint':name+'/last','step':2}]}
                return {'checkpoint':name}
            def evaluate(self,name,parent):
                if name=='initial':return dict(BASE)
                if '/S4/' in name:return {**BASE,'primary_asr':.6}
                if '/full/' in name and parent.endswith('/last'):
                    return {**BASE,'primary_asr':.1,'instruction_constraint':.8}
                return {**BASE,'primary_asr':.3}
        b=Checkpoints(c);result=Controller(c,b).run()
        grpo=[v for v in b.calls if '/S4/' in v[0]]
        self.assertEqual(len(grpo),c['search']['max_refinements_per_phase']+1)
        self.assertTrue(all(v[2].endswith('/trial-0/early') for v in grpo))
        self.assertTrue(result['retained_model'].endswith('/trial-0/early'))
        self.assertEqual(len(result['history'][0]['full_trials']),1+len(grpo))

    def test_positive_general_share_never_rounds_to_zero(self):
        b=object.__new__(Backend);b.c={'seed':42}
        rows=b.mix([{'id':'a'}],[],[{'id':'g'}],{'current':.85,'replay':0.,'general':.15})
        self.assertEqual([r['id'] for r in rows],['a','g'])
        self.assertEqual(b.last_mixture_audit['sampled_rows']['general'],1)
        with self.assertRaisesRegex(ValueError,'unavailable'):
            b.mix([{'id':'a'}],[],[],{'current':.85,'replay':0.,'general':.15})

if __name__=='__main__':unittest.main()
