import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from r2ai_autoresearch.research.data import prepare,split_safety_pool,final_panel
from r2ai_autoresearch.research.judging import judge
from r2ai_autoresearch.research.report import report
from r2ai_autoresearch.portable.benchmarks import IFEVAL_REVISION

class DataEvalTests(unittest.TestCase):
    def setUp(self):
        self.safety={'rows':[{'id':str(i),'kind':'harmful','prompt':'Unique safety request '+str(i)} for i in range(8)]}
        self.panels=[{'id':'hc','benchmark':'h_cot','kind':'harmful','prompt':'HC prompt'},
                     {'id':'xs','benchmark':'xstest','kind':'benign','prompt':'XS prompt'},
                     {'id':'if','benchmark':'ifeval','kind':'utility','prompt':'IF prompt','key':1,'instruction_id_list':['length_constraints:number_words'],'kwargs':[{'num_words':10,'relation':'less than'}]},
                     {'id':'wjbtest','benchmark':'wildjailbreak','kind':'harmful','prompt':'Final WJB test prompt'}]
        self.counts={k:1 for k in ['wildjailbreak','h_cot','xstest','ifeval']}
    def test_preparation_uses_pool_holdout_and_excludes_all_test_prompts(self):
        heldout,remaining=split_safety_pool(self.safety,1,42)
        train={'rows':[{'id':'attack','seed_id':remaining[0]['id'],'kind':'harmful','source':'attack','split':'train','prompt':'A transformed training request'}]}
        d=prepare(train,self.safety,self.panels,{},self.counts)
        primary=[r for r in d['rows'] if r.get('benchmark')=='wildjailbreak']
        self.assertEqual(primary[0]['prompt'],heldout[0]['prompt'])
        self.assertNotEqual(primary[0]['prompt'],'Final WJB test prompt')
        train['rows'][0]['seed_id']=heldout[0]['id']
        with self.assertRaisesRegex(ValueError,'held-out seed'):prepare(train,self.safety,self.panels,{},self.counts)
    def test_test_prompt_leakage_rejected(self):
        _,remaining=split_safety_pool(self.safety,1,42)
        train={'rows':[{'id':'attack','seed_id':remaining[0]['id'],'kind':'harmful','source':'attack','split':'train','prompt':'Final WJB test prompt'}]}
        with self.assertRaisesRegex(ValueError,'overlaps'):prepare(train,self.safety,self.panels,{},self.counts)
    def test_final_panel_has_no_training_rows(self):
        d=final_panel(self.panels,{})
        self.assertEqual(d['benchmarks']['usage'],'final_evaluation')
        self.assertTrue(all(r['split']=='dev' for r in d['rows']))
        self.assertEqual(d['benchmarks']['ifeval_revision'],IFEVAL_REVISION)
    def test_cli_prepares_custom_general_pool_and_receipt(self):
        from r2ai_autoresearch.research.data import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
            _,remaining=split_safety_pool(self.safety,1,42)
            train={'rows':[{'id':'attack','seed_id':remaining[0]['id'],'kind':'harmful','source':'darwin','split':'train','prompt':'Unique attack input'}]}
            (root/'train.json').write_text(json.dumps(train));(root/'pool.json').write_text(json.dumps(self.safety))
            (root/'general.json').write_text(json.dumps([
                {'prompt':'IF prompt','target':'Never train on this'},
                {'prompt':'General question','target':'Supervised answer'}]))
            c['data']['path']='prepared.json'
            prep=c['data']['preparation'];prep.update(training='train.json',safety_pool='pool.json',validation_counts=self.counts)
            prep['general'].update(source='custom',path='general.json')
            (root/'config.json').write_text(json.dumps(c))
            with patch('sys.argv',['prepare','--config',str(root/'config.json')]), patch('r2ai_autoresearch.research.data.load_suite',return_value=(self.panels,{})), patch('builtins.print'):
                main()
            result=json.loads((root/'prepared.json').read_text())
            general=[r for r in result['rows'] if r['kind']=='utility' and r['split']=='train']
            self.assertEqual([r['target'] for r in general],['Supervised answer'])
            self.assertEqual(result['benchmarks']['sources']['general_data']['skipped']['excluded_prompt'],1)
    def test_score5_is_unsafe_score4_is_not(self):
        class Client:judge_concurrency=2
        rows=[{'id':str(i),'kind':'harmful','prompt':'p'} for i in range(2)]
        def fake(c,batch,group,rubric):
            id=batch[0][0]['id'];return {id:{'id':id,'score':4+int(id),'reason':'fixture'}}
        with patch('r2ai_autoresearch.research.judging.judge_batch',side_effect=fake):
            result=judge(Client(),rows,['response','response'])
        self.assertFalse(result[0]['unsafe']);self.assertTrue(result[1]['unsafe'])
    def test_report_failure_is_blank_not_zero_metric(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s={'retained_model':'M0','history':[], 'trials':[{'id':'trial','phase':'pilot','recipe':{'source':'a','method':'direct_grpo','sample_count':1},'metrics':None,'gate':{'eligible':False,'safety_gain':-1,'issues':['training_signal']},'parent':'M0','checkpoint':None}]}
            (p/'summary.json').write_text(json.dumps(s));report(p)
            import csv
            with (p/'report/trials.csv').open() as f:row=list(csv.DictReader(f))[0]
            self.assertEqual(row['primary_asr'],'');self.assertEqual(row['safety_gain'],'')
