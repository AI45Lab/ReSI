import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from r2ai_autoresearch.research.generation import evaluation_outputs
from r2ai_autoresearch.research.config import load, validate
from r2ai_autoresearch.portable.loop import Loop
from r2ai_autoresearch.portable.errors import CandidateEvaluationError


class FakeLoop:
    def __init__(self, root, retry_truncated=False):
        self.root=Path(root);self.client=object();self.calls=[]
        self.dev=[{'id':'a'},{'id':'b'}]
        self.c={'role':'benchmark_development','max_new_tokens':2048,'evaluation':{
            'max_new_tokens':32768,'retry_max_new_tokens':65536,'truncation_policy':'error',
            'generation':{'temperature':None}}}
        self.initial=[{'id':'a','text':'Completed answer','hit_token_limit':False},
                      {'id':'b','text':'Partial answer','hit_token_limit':True}]
        self.retry=[{'id':'b','text':'Extended answer','hit_token_limit':retry_truncated}]

    def worker(self,name,operation,parent,rows):
        self.calls.append({'name':name,'ids':[r['id'] for r in rows], 'config':copy.deepcopy(self.c)})
        return {'outputs':copy.deepcopy(self.retry if name.endswith('generation_retry') else self.initial),
                'generation_config':{'temperature':0.6,'top_p':0.95,'do_sample':True}}


class EvaluationRetryTests(unittest.TestCase):
    def test_only_truncated_items_retried_and_original_config_preserved(self):
        loop=FakeLoop('.');before=copy.deepcopy(loop.c)
        outputs,receipt=evaluation_outputs(loop,'target','M',loop.dev)
        self.assertEqual([r['text'] for r in outputs],['Completed answer','Extended answer'])
        self.assertEqual(loop.calls[1]['ids'],['b'])
        self.assertEqual(loop.calls[1]['config']['evaluation']['max_new_tokens'],65536)
        self.assertEqual(loop.calls[0]['config']['evaluation']['max_new_tokens'],32768)
        self.assertEqual(loop.c,before)
        self.assertEqual(receipt['retried_ids'],['b'])
        self.assertEqual(receipt['remaining_truncated_ids'],[])

    def test_no_retry_for_complete_outputs_or_disabled_retry(self):
        loop=FakeLoop('.');loop.initial[1]['hit_token_limit']=False
        evaluation_outputs(loop,'target','M',loop.dev)
        self.assertEqual(len(loop.calls),1)
        loop=FakeLoop('.');loop.c['evaluation']['retry_max_new_tokens']=None
        _,receipt=evaluation_outputs(loop,'target','M',loop.dev)
        self.assertEqual(len(loop.calls),1)
        self.assertEqual(receipt['remaining_truncated_ids'],['b'])

    def test_remaining_truncation_blocks_scoring_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            loop=FakeLoop(d,retry_truncated=True)
            with patch('r2ai_autoresearch.portable.benchmarks.score') as score:
                with self.assertRaises(CandidateEvaluationError):Loop.evaluate(loop,'target','M')
                score.assert_not_called()
            receipt=json.loads((Path(d)/'target/generation_receipt.json').read_text())
            self.assertEqual(receipt['remaining_truncated_ids'],['b'])
            self.assertEqual(len(loop.calls),2)

    def test_score_only_selected_final_responses(self):
        with tempfile.TemporaryDirectory() as d:
            loop=FakeLoop(d)
            with patch('r2ai_autoresearch.portable.benchmarks.score',return_value={'metrics':{},'counts':{}}) as score:
                record=Loop.evaluate(loop,'target','M')
                self.assertEqual([o['text'] for o in score.call_args.args[2]],['Completed answer','Extended answer'])
            self.assertEqual(record['generation']['retried_ids'],['b'])

    def test_retry_ids_must_match_and_missing_outputs_are_not_safe(self):
        loop=FakeLoop('.');loop.retry=[]
        with self.assertRaisesRegex(ValueError,'retry generation coverage'):evaluation_outputs(loop,'target','M',loop.dev)

    def test_caps_match_appendix_and_invalid_retry_rejected(self):
        c=load(Path(__file__).resolve().parents[1]/'config.json')
        self.assertEqual(c['evaluation']['max_new_tokens'],32768)
        self.assertEqual(c['evaluation']['retry_max_new_tokens'],65536)
        c['evaluation']['retry_max_new_tokens']=32768
        with self.assertRaises(ValueError):validate(c)

    def test_missing_final_is_diagnosed_before_judging(self):
        for answer in ('', '<think>still reasoning'):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as d:
                loop=FakeLoop(d)
                loop.initial[0].update(text=answer,hit_token_limit=False)
                with patch('r2ai_autoresearch.portable.benchmarks.score') as score:
                    with self.assertRaises(CandidateEvaluationError):Loop.evaluate(loop,'target','M')
                    score.assert_not_called()
                diagnostic=json.loads((Path(d)/'target/evaluation_diagnostics.json').read_text())
                self.assertEqual(diagnostic['status'],'blocked')
                self.assertIn('a',diagnostic['issues']['empty_final'])

    def test_completed_format_violation_remains_scoreable_and_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            loop=FakeLoop(d)
            loop.dev[0].update(benchmark='ifeval',prompt='Return valid JSON.')
            loop.initial[0]['text']='This is not JSON.'
            with patch('r2ai_autoresearch.portable.benchmarks.score',return_value={'metrics':{'instruction_constraint':0.},'counts':{}}) as score:
                record=Loop.evaluate(loop,'target','M')
            self.assertEqual(record['metrics']['instruction_constraint'],0.)
            self.assertEqual(score.call_args.args[2][0]['text'],'This is not JSON.')
            self.assertEqual(record['diagnostics']['status'],'ready')

    def test_explicit_final_incomplete_flag_blocks_even_with_nonempty_text(self):
        from r2ai_autoresearch.research.diagnostics import response_diagnostics
        diagnostic=response_diagnostics([{'id':'a'}],[{'id':'a','text':'Partial reasoning','final_complete':False,'hit_token_limit':False}])
        self.assertEqual(diagnostic['blocking_issues'],['incomplete_final'])

    def test_score_and_report_does_not_hide_truncation(self):
        from r2ai_autoresearch.research.diagnostics import response_diagnostics
        diagnostic=response_diagnostics([{'id':'a','benchmark':'ifeval'}],[{'id':'a','text':'Partial answer','hit_token_limit':True}],'score_and_report')
        self.assertEqual(diagnostic['status'],'score_and_report')
        self.assertEqual(diagnostic['panels']['ifeval']['truncated'],1)

    def test_frequent_failures_require_training_review_with_initial_comparison(self):
        from r2ai_autoresearch.research.diagnostics import response_diagnostics
        rows=[{'id':str(i),'benchmark':'ifeval'} for i in range(10)]
        good=[{'id':str(i),'text':'Completed answer','hit_token_limit':False} for i in range(10)]
        baseline=response_diagnostics(rows,good)
        broken=copy.deepcopy(good)
        for o in broken[:3]:o.update(text='<think>unfinished',final_complete=False)
        d=response_diagnostics(rows,broken,baseline=baseline)
        self.assertTrue(d['training_review_required'])
        self.assertEqual(d['abnormal_count'],3) # Multiple flags on one item count once.
        self.assertEqual(d['baseline_comparison']['newly_abnormal_ids'],['0','1','2'])
        self.assertIn('label masking',d['recommended_checks'][0])
        self.assertFalse(response_diagnostics(rows,broken,settings={'training_review_min_count':4,'training_review_fraction':.2})['training_review_required'])

    def test_panel_specific_failure_is_not_diluted_by_other_benchmarks(self):
        from r2ai_autoresearch.research.diagnostics import response_diagnostics
        rows=[{'id':str(i),'benchmark':'ifeval' if i<10 else 'wildjailbreak'} for i in range(100)]
        outputs=[{'id':str(i),'text':'' if i<3 else 'Answer','hit_token_limit':False} for i in range(100)]
        self.assertTrue(response_diagnostics(rows,outputs)['training_review_required'])

    def test_supervision_audit_exposes_masked_target_and_missing_eos(self):
        from r2ai_autoresearch.research.diagnostics import supervision_diagnostics
        d=supervision_diagnostics([('good',[1,2,9],[-100,2,9]),('masked',[1,9],[-100,-100]),('no-eos',[1,2],[-100,2])],9)
        self.assertEqual(d['zero_supervision_ids'],['masked'])
        self.assertEqual(d['missing_supervised_terminal_eos_ids'],['masked','no-eos'])
        self.assertEqual(d['rows'][0]['supervised_tokens'],2)


if __name__=='__main__':unittest.main()
