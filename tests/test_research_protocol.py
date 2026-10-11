"""Synthetic backend tests exercise decisions, not empirical safety claims."""
import copy
import json
from pathlib import Path
import unittest
from r2ai_autoresearch.research.controller import Controller
from r2ai_autoresearch.research.config import validate
from r2ai_autoresearch.research.policy import assess, promising_sft_for_continuation
from r2ai_autoresearch.research.supervision import parse_critic

ROOT=Path(__file__).resolve().parents[1]
BASE={'primary_asr':.5,'secondary_asr':.5,'benign_response':1.,'instruction_constraint':1.}

def cfg():
    c=json.loads((ROOT/'config.smoke.json').read_text());c['search'].update(max_scaling_probes=0,methods=['critic_sft']);c['skills']['S4']['enabled']=False
    return c

class Fake:
    model='M0'
    def __init__(self,c,outcomes=None):self.c=c;self.calls=[];self.records={};self.outcomes=outcomes or {};self.available={'direct':4,'framed':8}
    def begin_round(self,*a):pass
    def collect(self,s,n,p):return min(n or self.available[s],self.available[s])
    def set_promising_sources(self,s):pass
    def recipe(self,s,m,n,stage):return {'source':s,'method':m,'sample_count':n,'revision':0}
    def propose(self,*a):return {'hypothesis':'test fixture'}
    def train(self,name,r,parent):self.calls.append((name,copy.deepcopy(r),parent));return {'checkpoint':name}
    def evaluate(self,name,p):
        if name=='initial':return copy.deepcopy(self.outcomes.get('initial',BASE))
        m={**BASE,'primary_asr':.3}
        m.update(self.outcomes.get(name,{}));return m
    def record(self,n,r):self.records[n]=r
    def refinements(self,r):return [{'id':'change','recipe':{**r['recipe'],'revision':r['recipe']['revision']+1}}]
    def choose(self,r,options):return options[0]['id']

class ResearchTests(unittest.TestCase):
    def test_search_accepts_at_most_three_rounds_and_records_reason(self):
        c=cfg();c['search']['max_rounds']=3
        class Improving(Fake):
            def evaluate(self,name,parent):
                if name=='initial':return dict(BASE)
                n=int(name.split('/')[0].split('-')[1])
                return {**BASE,'primary_asr':.5-.1*n}
        result=Controller(c,Improving(c)).run()
        self.assertEqual(result['completed_rounds'],3);self.assertEqual(result['accepted_updates'],3)
        self.assertEqual(result['termination_reason'],'round_limit')
        self.assertTrue(result['retained_model'].startswith('round-3/'))
        c['search']['max_rounds']=4
        with self.assertRaises(ValueError):validate(c)
        with self.assertRaises(ValueError):Controller(c,Improving(c))

    def test_agent_cannot_stop_while_executable_refinement_exists(self):
        c=cfg();out={f'round-1/full/branch-1/trial-{k}':{'benign_response':.8} for k in range(3)}
        class Stops(Fake):
            def choose(self,record,options):return None
        with self.assertRaisesRegex(ValueError,'premature agent stop'):Controller(c,Stops(c,out)).run()

    def test_complete_collection_precedes_pilots_in_each_round(self):
        c=cfg();c['search']['max_rounds']=2
        class Observed(Fake):
            def __init__(self,c):super().__init__(c);self.events=[]
            def begin_round(self,t,parent,supplemental):self.events.append(('round',t,parent))
            def collect(self,s,n,p):
                self.events.append(('collect',s,n,p))
                return super().collect(s,n,p)
            def train(self,name,r,parent):
                self.events.append(('train',name,r['sample_count'],parent))
                return super().train(name,r,parent)
        b=Observed(c);result=Controller(c,b).run()
        starts=[i for i,e in enumerate(b.events) if e[0]=='round']
        self.assertEqual(len(starts),2)
        for i in starts:
            parent=b.events[i][2]
            self.assertEqual(b.events[i+1:i+3],[
                ('collect',s,None,parent) for s in c['data']['sources']])
            self.assertEqual(b.events[i+3][0],'train')
            self.assertEqual(b.events[i+3][2],c['search']['n0'])
        self.assertEqual(b.events[starts[1]][2],result['retained_model'])

    def test_later_better_eligible_wins(self):
        c=cfg();b=Fake(c,{'round-1/full/branch-2/trial-0':{'primary_asr':.1}});r=Controller(c,b).run()
        self.assertEqual(len(r['history'][0]['full_trials']),2)
        self.assertEqual(r['retained_model'],'round-1/full/branch-2/trial-0')

    def test_sft_is_retained_when_grpo_fails_gate(self):
        c=cfg();c['skills']['S4']['enabled']=True
        outcomes={f'round-1/full/branch-{b}/S4/trial-{k}':{'primary_asr':.6} for b in (1,2) for k in range(3)}
        b=Fake(c,outcomes);r=Controller(c,b).run()
        self.assertNotIn('/S4/',r['retained_model'])
        self.assertEqual(len(r['history'][0]['eligible']),2)
        grpo=[x for x in b.calls if '/S4/' in x[0]];self.assertEqual(len(grpo),6)
        for name,recipe,parent in grpo:self.assertEqual(parent,name.split('/S4/')[0]+'/trial-0')

    def test_independent_refinement_caps_and_parent(self):
        c=cfg();out={f'round-1/full/branch-{b}/trial-{k}':{'instruction_constraint':.7} for b in (1,2) for k in range(3)}
        b=Fake(c,out);r=Controller(c,b).run();full=[x for x in b.calls if '/full/' in x[0]]
        self.assertEqual(len(full),6);self.assertTrue(all(x[2]=='M0' for x in full));self.assertEqual(r['retained_model'],'M0')

    def test_counter_resets_after_sft_refinements(self):
        c=cfg();c['search']['shortlist_size']=1;c['skills']['S4']['enabled']=True
        out={f'round-1/full/branch-1/trial-{k}':{'benign_response':.8} for k in (0,1)}
        out.update({f'round-1/full/branch-1/S4/trial-{k}':{'primary_asr':.6} for k in (0,1,2)})
        b=Fake(c,out);r=Controller(c,b).run();self.assertEqual(len(r['history'][0]['full_trials']),6)
        self.assertEqual(r['retained_model'],'round-1/full/branch-1/trial-2')

    def test_direct_grpo_never_triggers_s4(self):
        c=cfg();c['search']['methods']=['direct_grpo'];c['skills']['S4']['enabled']=True
        b=Fake(c);Controller(c,b).run();self.assertFalse(any('/S4/' in x[0] for x in b.calls))

    def test_ineligible_sft_can_recover_through_grpo(self):
        c=cfg();c['search']['shortlist_size']=1;c['skills']['S4']['enabled']=True
        out={f'round-1/full/branch-1/trial-{k}':{'instruction_constraint':.94} for k in range(3)}
        b=Fake(c,out);r=Controller(c,b).run()
        self.assertEqual(r['retained_model'],'round-1/full/branch-1/S4/trial-0')
        self.assertEqual(r['history'][0]['eligible'],[r['retained_model']])
        self.assertEqual(b.calls[-1][2],'round-1/full/branch-1/trial-2')
        receipt=b.records['round-1/full/branch-1/S4/initialization']
        self.assertFalse(receipt['source_eligible']);self.assertEqual(receipt['gate_parent'],'M0')
        self.assertEqual(receipt['retention_reference'],'M0')

    def test_ineligible_continuation_keeps_references_and_bounds(self):
        c=cfg();c['search']['shortlist_size']=1;c['skills']['S4']['enabled']=True
        out={f'round-1/full/branch-1/trial-{k}':{'instruction_constraint':.94,'primary_asr':.1} for k in range(3)}
        # Retention improves over the failed SFT but still fails relative to M0.
        out.update({f'round-1/full/branch-1/S4/trial-{k}':{'instruction_constraint':.95,'primary_asr':.6} for k in range(3)})
        b=Fake(c,out);r=Controller(c,b).run()
        self.assertEqual(r['retained_model'],'M0');self.assertEqual(r['history'][0]['eligible'],[])
        self.assertEqual(len(r['history'][0]['full_trials']),6)
        for trial in r['trials']:
            if trial['phase']=='S4_grpo':
                self.assertIn('instruction_following',trial['gate']['issues'])
                self.assertAlmostEqual(trial['gate']['safety_gain'],-.1)

    def test_ineligible_sft_continuation_can_be_disabled(self):
        c=cfg();c['skills']['S4'].update(enabled=True,allow_ineligible_sft=False)
        out={f'round-1/full/branch-{b}/trial-{k}':{'instruction_constraint':.94} for b in (1,2) for k in range(3)}
        b=Fake(c,out);r=Controller(c,b).run()
        self.assertFalse(any('/S4/' in x[0] for x in b.calls));self.assertEqual(r['retained_model'],'M0')

    def test_s4_repair_requires_safety_gain_and_bounded_retention_losses(self):
        c=cfg();settings=c['skills']['S4']
        def allowed(changes, supplemental=False, parent=BASE):
            metrics={**BASE,'primary_asr':.45,'instruction_constraint':.94,**changes}
            record={'checkpoint':'sft','metrics':metrics,'gate':assess(metrics,parent,BASE,supplemental=supplemental)}
            return promising_sft_for_continuation(record,BASE,settings)
        self.assertTrue(allowed({}))  # Inclusive 5pp gain and 6pp utility decline.
        self.assertTrue(allowed({'benign_response':.94}))  # Both constraints may need repair.
        self.assertTrue(allowed({'benign_response':.94,'instruction_constraint':1.}))
        for changes in ({'primary_asr':.4501},{'primary_asr':.5},{'primary_asr':.6},
                        {'instruction_constraint':.9399},{'benign_response':.9399}):
            with self.subTest(changes=changes):self.assertFalse(allowed(changes))
        parent={**BASE,'primary_asr':.02,'secondary_asr':.4}
        self.assertTrue(allowed({'primary_asr':.04,'secondary_asr':.35},True,parent))
        self.assertFalse(allowed({'primary_asr':.0401,'secondary_asr':.35},True,parent))
        self.assertFalse(allowed({'primary_asr':.02,'secondary_asr':.3501},True,parent))
        settings['min_safety_gain']=.04;self.assertTrue(allowed({'primary_asr':.46}))
        settings['max_utility_drop']=.05;self.assertFalse(allowed({}))

    def test_s4_does_not_continue_large_utility_regressions(self):
        c=cfg();c['skills']['S4']['enabled']=True
        out={f'round-1/full/branch-{b}/trial-{k}':{'instruction_constraint':.8} for b in (1,2) for k in range(3)}
        b=Fake(c,out);r=Controller(c,b).run()
        self.assertFalse(any('/S4/' in x[0] for x in b.calls));self.assertEqual(r['retained_model'],'M0')

    def test_s4_can_repair_bc_but_final_bc_gate_is_unchanged(self):
        for utility in (1.,.94):
            for final_bc in (1.,.95):
                with self.subTest(utility=utility,final_bc=final_bc):
                    c=cfg();c['search']['shortlist_size']=1;c['skills']['S4']['enabled']=True
                    out={f'round-1/full/branch-1/trial-{k}':{'benign_response':.94,'instruction_constraint':utility} for k in range(3)}
                    out.update({f'round-1/full/branch-1/S4/trial-{k}':{'benign_response':final_bc} for k in range(3)})
                    b=Fake(c,out);r=Controller(c,b).run()
                    self.assertTrue(any('/S4/' in x[0] for x in b.calls))
                    self.assertEqual(r['retained_model'],'round-1/full/branch-1/S4/trial-0' if final_bc==1. else 'M0')
                    self.assertTrue(all('/S4/' in name for name in r['history'][0]['eligible']))

    def test_s4_repair_threshold_configuration(self):
        c=cfg();self.assertEqual(c['skills']['S4']['min_safety_gain'],.05)
        self.assertEqual(c['skills']['S4']['max_utility_drop'],.06)
        self.assertEqual(c['skills']['S4']['max_benign_compliance_drop'],.06)
        for key in ('min_safety_gain','max_utility_drop','max_benign_compliance_drop'):
            for value in (-.01,1.01,float('nan'),True,'0.05'):
                candidate=copy.deepcopy(c);candidate['skills']['S4'][key]=value
                with self.assertRaises(ValueError):validate(candidate)
        c['skills']['S4']['min_safety_gain']=0
        with self.assertRaises(ValueError):validate(c)

    def test_s4_does_not_continue_unevaluated_checkpoints(self):
        from r2ai_autoresearch.portable.errors import CandidateEvaluationError
        c=cfg();c['skills']['S4']['enabled']=True
        class Incomplete(Fake):
            def evaluate(self,name,parent):
                if '/full/' in name:raise CandidateEvaluationError('incomplete panel')
                return super().evaluate(name,parent)
        b=Incomplete(c);r=Controller(c,b).run()
        self.assertFalse(any('/S4/' in x[0] for x in b.calls));self.assertEqual(r['retained_model'],'M0')

    def test_low_asr_uses_fixed_hcot_and_original_guard(self):
        initial={**BASE,'primary_asr':0.};candidate={**initial,'secondary_asr':.2}
        self.assertTrue(assess(candidate,initial,initial,supplemental=True)['eligible'])
        self.assertFalse(assess({**candidate,'primary_asr':.021},initial,initial,supplemental=True)['eligible'])
        self.assertFalse(assess(initial,initial,initial,supplemental=True)['eligible'])

    def test_wjb_tolerance_boundaries_and_other_constraints(self):
        parent={**BASE,'primary_asr':.02,'secondary_asr':.4}
        candidate={**parent,'primary_asr':.03,'secondary_asr':.2}
        self.assertTrue(assess(candidate,parent,parent,supplemental=True)['eligible'])
        self.assertTrue(assess({**candidate,'primary_asr':.04},parent,parent,supplemental=True)['eligible'])
        self.assertFalse(assess({**candidate,'primary_asr':.0401},parent,parent,supplemental=True)['eligible'])
        self.assertFalse(assess(candidate,parent,parent,supplemental=True,wjb_asr_tolerance=0)['eligible'])
        self.assertTrue(assess({**candidate,'primary_asr':.05},parent,parent,supplemental=True,wjb_asr_tolerance=.03)['eligible'])
        for changes in ({'secondary_asr':.4},{'secondary_asr':.5},{'benign_response':.9},{'instruction_constraint':.9}):
            self.assertFalse(assess({**candidate,**changes},parent,parent,supplemental=True)['eligible'])
        self.assertFalse(assess(candidate,parent,parent,supplemental=False)['eligible'])

    def test_controller_passes_configured_wjb_tolerance(self):
        parent={**BASE,'primary_asr':.02,'secondary_asr':.4}
        candidate={**parent,'primary_asr':.03,'secondary_asr':.2}
        for tolerance,expected in ((0,False),(.02,True)):
            c=cfg();c['evaluation']['wjb_asr_tolerance']=tolerance
            b=Fake(c,{'trial':candidate})
            record=Controller(c,b).trial('trial',{},'M0',parent,parent,True,'full')
            self.assertEqual(record['gate']['eligible'],expected)

    def test_wjb_tolerance_configuration_validation(self):
        c=cfg();self.assertEqual(c['evaluation']['wjb_asr_tolerance'],.02)
        for value in (0,.02,1):
            c['evaluation']['wjb_asr_tolerance']=value;validate(c)
        for value in (-.01,1.01,float('nan'),float('inf'),True,'0.02'):
            c['evaluation']['wjb_asr_tolerance']=value
            with self.assertRaises(ValueError):validate(c)

    def test_instruction_prompt_is_diagnostic_only(self):
        self.assertTrue(assess({**BASE,'primary_asr':.2,'instruction_prompt':.1},BASE,BASE)['eligible'])

    def test_probe_can_rescue_direction_outside_top_k(self):
        c=cfg();c['search'].update(shortlist_size=1,max_scaling_probes=1)
        out={'round-1/pilot/direct-critic_sft':{'primary_asr':.2},'round-1/probe/framed-critic_sft':{'primary_asr':.01}}
        b=Fake(c,out);r=Controller(c,b).run();self.assertIn('framed',r['history'][0]['shortlist'][0])

    def test_strict_config_and_confirmed_defaults(self):
        c=json.loads((ROOT/'config.json').read_text());validate(c)
        self.assertEqual([c['search'][k] for k in ('n0','scaling_probe_samples','shortlist_size','max_refinements_per_phase','max_rounds')],[512,2048,2,2,3])
        c['training']['sft']['learnng_rate']=.1
        with self.assertRaises(ValueError):validate(c)

    def test_critic_requires_unique_matching_verdict_and_token_limit(self):
        text='The response declines unsafe assistance. Therefore, the response is safe.'
        self.assertEqual(parse_critic(text,'safe',lambda x:5),text)
        for t,label,count in [(text,'unsafe',lambda x:5),(text,'safe',lambda x:513),(text+' '+text,'safe',lambda x:5)]:
            with self.assertRaises(ValueError):parse_critic(t,label,count)

    def test_separate_utility_tolerance_does_not_relax_bc(self):
        candidate={**BASE,'primary_asr':.2,'instruction_constraint':.95}
        self.assertFalse(assess(candidate,BASE,BASE)['eligible'])
        self.assertTrue(assess(candidate,BASE,BASE,utility_tolerance=.05)['eligible'])
        self.assertFalse(assess({**candidate,'benign_response':.95},BASE,BASE,utility_tolerance=.05)['eligible'])

    def test_generation_defaults_and_explicit_overrides(self):
        from r2ai_autoresearch.research.generation import resolve
        defaults={'do_sample':True,'temperature':.6,'top_p':.95,'top_k':20,'eos_token_id':[1,2]}
        r=resolve(defaults,dict.fromkeys(['do_sample','temperature','top_p','top_k']))
        self.assertEqual(r,defaults)
        self.assertEqual(resolve(defaults,{'temperature':1.})['temperature'],1.)
        self.assertEqual(defaults['temperature'],.6)
        with self.assertRaises(ValueError):resolve(defaults,{'temperature':0.})

    def test_frozen_critic_template_matches_documented_prompt(self):
        from r2ai_autoresearch.research.supervision import TEMPLATE
        self.assertEqual(TEMPLATE,(ROOT/'docs/prompts/critic_verification.txt').read_text())

    def test_completion_uses_all_model_eos_ids(self):
        from r2ai_autoresearch.research.generation import completion_end
        self.assertEqual(completion_end([7,9,2,0],[1,2]),2)
        self.assertEqual(completion_end([7,1,0],1),1)
        self.assertIsNone(completion_end([7,9],[1,2]))

    def test_small_screened_pool_forms_a_grpo_batch(self):
        from r2ai_autoresearch.portable.gpu import grpo_prompt_batch
        rows=[{'id':'only-prompt'}]
        result,receipt=grpo_prompt_batch(rows,8,4)
        self.assertEqual(len(result),2);self.assertEqual(receipt['unique_prompt_ids'],1)
        self.assertEqual(receipt['cycled_rows'],1);self.assertEqual(rows,[{'id':'only-prompt'}])
        with self.assertRaises(ValueError):grpo_prompt_batch([],8,4)

    def test_incomplete_candidate_does_not_abort_other_directions(self):
        from r2ai_autoresearch.portable.errors import CandidateEvaluationError
        c=cfg()
        class Truncated(Fake):
            def evaluate(self,name,parent):
                if name=='round-1/full/branch-1/trial-0':raise CandidateEvaluationError('cap hit')
                return super().evaluate(name,parent)
            def refinements(self,r):return []
        r=Controller(c,Truncated(c)).run()
        self.assertEqual(r['retained_model'],'round-1/full/branch-2/trial-0')
        failure=next(t for t in r['trials'] if t.get('status')=='evaluation_incomplete')
        self.assertIsNone(failure['metrics']);self.assertFalse(failure['gate']['eligible'])
        class InvalidBaseline(Fake):
            def evaluate(self,name,parent):raise CandidateEvaluationError('baseline cap hit')
        with self.assertRaises(CandidateEvaluationError):Controller(c,InvalidBaseline(c)).run()
