import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from r2ai_autoresearch.formal_eval.api import generate_manifest
from r2ai_autoresearch.formal_eval.config import Endpoint,GenerationConfig
from r2ai_autoresearch.formal_eval.scoring import prepare_judge,parse_safety_judge

class FormalEvaluationTests(unittest.TestCase):
    def test_formal_request_limits_follow_unified_configuration(self):
        from r2ai_autoresearch.research.full_evaluate import prepare, attack_roles, default_settings
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config.smoke.json').read_text())
        cfg['final_evaluation']['target'].update(base_url='http://localhost:18080/v1',model='fixture')
        cfg['final_evaluation']['requests'].update(target_timeout_seconds=300,max_retries=0,transient_retries=1)
        cfg['services']['judge']['timeout_seconds']=17
        with tempfile.TemporaryDirectory() as folder:
            cfg['model']['path']=folder
            with patch('r2ai_autoresearch.research.full_evaluate.Client.endpoint',return_value=('http://example.test/v1','judge','')):
                target,judge,_,_,_=prepare(cfg,Path(folder)/'config.json')
        self.assertEqual(target['timeout_seconds'],300)
        self.assertEqual(judge['timeout_seconds'],17)
        self.assertEqual(judge['max_retries'],0)
        self.assertEqual(judge['transient_retries'],1)
        roles=attack_roles(default_settings()['xteaming'],{**judge,'timeout_seconds':23},judge,target,{'max_tokens':32},{},'')
        self.assertEqual(roles['attacker']['timeout_seconds'],23)
        self.assertEqual(roles['planner']['transient_retries'],1)
        self.assertEqual(roles['verifier']['timeout_seconds'],17)

    def test_xteaming_defaults_match_paper_roles_and_inherit_service(self):
        from r2ai_autoresearch.research.full_evaluate import default_settings,attack_roles
        settings=default_settings()['xteaming']
        inherited={'base_url':'http://example.test/v1','model':'gpt-4o','api_key_env':'INHERITED_KEY','temperature':1.,'top_p':.95,'max_tokens':4096}
        roles=attack_roles(settings,inherited,{'model':'gpt-4o'},{'model':'target'}, {'max_tokens':32768},{},'native system')
        self.assertEqual(roles['planner']['model'],'DeepSeek-R1')
        self.assertEqual(roles['attacker']['model'],'Qwen2.5-32B-Instruct')
        self.assertEqual(roles['optimizer'],roles['attacker'])
        self.assertEqual(roles['attacker']['api_key_env'],'INHERITED_KEY')
        self.assertEqual(roles['target']['system_prompt'],'native system')

    def test_xteaming_roles_are_independently_configurable(self):
        from r2ai_autoresearch.research.config import validate
        cfg=json.loads((Path(__file__).resolve().parents[1]/"config.smoke.json").read_text())
        xt=cfg["final_evaluation"]["xteaming"]
        for role,model in (("planner","DeepSeek-R1"),("attacker","Qwen2.5-32B-Instruct")):
            xt[role]={"base_url":"http://localhost:18000/v1","model":model,"api_key_env":"TEST_KEY"}
        validate(cfg)
        self.assertNotEqual(xt["planner"]["model"],xt["attacker"]["model"])
        xt["max_textgrad_iterations"]=0
        with self.assertRaises(ValueError):validate(cfg)

    def test_impossible_xteaming_plan_count_fails_config_preflight(self):
        from r2ai_autoresearch.research.config import validate
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config.smoke.json').read_text())
        cfg['final_evaluation']['xteaming'].update(planner_sets=1,plans_per_set=1,plans_to_execute=2)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            validate(cfg)

    def test_ministral_formal_evaluation_preserves_native_system_prompt(self):
        from r2ai_autoresearch.research.full_evaluate import prepare
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config.smoke.json').read_text())
        cfg['final_evaluation']['target'].update(base_url='http://127.0.0.1:18080/v1',model='fixture')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg['model']['path']=folder
            (root/'config.json').write_text('{"model_type":"mistral3"}')
            (root/'SYSTEM_PROMPT.txt').write_text('Native system prompt')
            with patch('r2ai_autoresearch.research.full_evaluate.Client.endpoint',return_value=('http://example.test/v1','judge','')):
                prepare(cfg,root/'runtime.json')
                self.assertEqual(cfg['evaluation']['system_prompt'],'Native system prompt')
                cfg['evaluation']['system_prompt']='Different prompt'
                with self.assertRaisesRegex(ValueError,'native system prompt'):prepare(cfg,root/'runtime.json')

    def test_formal_generation_preserves_configured_thinking_mode(self):
        from r2ai_autoresearch.research.full_evaluate import prepare
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config.smoke.json').read_text())
        cfg['final_evaluation']['target'].update(base_url='http://127.0.0.1:18080/v1',model='fixture')
        with tempfile.TemporaryDirectory() as folder:
            cfg['model']['path']=folder
            (Path(folder)/'generation_config.json').write_text(json.dumps({'temperature':.7,'top_p':.9,'top_k':20}))
            for thinking in (False,True):
                cfg['model']['enable_thinking']=thinking
                with patch('r2ai_autoresearch.research.full_evaluate.Client.endpoint',return_value=('http://127.0.0.1:18081/v1','judge','')):
                    _,_,generation,extra,_=prepare(cfg,Path(folder)/'config.json')
                self.assertEqual(extra['chat_template_kwargs'],{'enable_thinking':thinking})
                self.assertEqual(generation['temperature'],.7)
                self.assertEqual(extra['top_k'],20)

    def test_generation_retries_with_extended_cap_and_rejects_persistent_truncation(self):
        for final_reason,failed in [('stop',0),('length',1)]:
            with self.subTest(final_reason=final_reason),tempfile.TemporaryDirectory() as folder:
                results=[{'response':'partial','finish_reason':'length'}, {'response':'final','finish_reason':final_reason}]
                path=Path(folder)/'out.jsonl'
                with patch('r2ai_autoresearch.formal_eval.api.chat_completion',side_effect=results) as call:
                    counts=generate_manifest([{'id':'a','benchmark':'wildjailbreak','prompt':'fixture'}],Endpoint('fixture'),path,
                        GenerationConfig(max_tokens=32,retry_max_tokens=64),workers=1,resume=True)
                self.assertEqual(counts['failed'],failed)
                self.assertEqual(call.call_args_list[1].args[2].max_tokens,64)
                record=json.loads(path.read_text());self.assertIn('extended_budget_retry',record)
                self.assertEqual('error' in record,bool(failed))

    def test_invalid_judge_is_failure_instead_of_reducing_the_denominator(self):
        from r2ai_autoresearch.formal_eval.scoring import judge_predictions
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pred=root/'predictions.jsonl'
            pred.write_text(json.dumps({'id':'a','response':'fixture final'})+'\n')
            with patch('r2ai_autoresearch.formal_eval.scoring.chat_completion',return_value={'response':'invalid','finish_reason':'stop'}):
                result=judge_predictions([{'id':'a','benchmark':'harmbench','prompt':'fixture'}],pred,Endpoint('fixture'),root/'judge.jsonl',evaluation_root=root,workers=1,resume=False)
            self.assertEqual(result['failed'],1);self.assertEqual(result['succeeded'],0)

    def test_prompt_bundled_without_old_dataset_directory(self):
        template,generation=prepare_judge('harmbench','/nonexistent/old/location')
        self.assertIn('{QUESTION}',template);self.assertIn('{RESPONSE}',template)
        self.assertEqual(generation.temperature,0);self.assertEqual(generation.max_tokens,1024)
        self.assertEqual(parse_safety_judge('{"score":5,"reason":"fixture"}')[0],5)

    def test_nested_metrics_preserve_instruction_following_in_csv(self):
        from r2ai_autoresearch.research.full_evaluate import numeric_metrics
        self.assertEqual(dict(numeric_metrics({'loose':{'instruction_level_accuracy':.8}})),{'loose.instruction_level_accuracy':.8})

    def test_endpoint_does_not_allow_external_python_proxy(self):
        with self.assertRaises(ValueError):Endpoint.from_mapping({'model':'x','transport':'file_api_proxy_v2'})
        with self.assertRaises(ValueError):Endpoint('x',transport='file_api_proxy_v2')

    def test_xteaming_protocol_uses_bundled_assets(self):
        try:
            from r2ai_autoresearch.formal_eval.xteaming import _load_protocol_prompts,_resolved_config
        except ImportError as e:self.skipTest('Install requirements-eval.txt: '+str(e))
        cfg={'roles':{'target':{'temperature':None,'max_tokens':32768}}}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prompts=_load_protocol_prompts(cfg,root/'config.json')
            resolved=_resolved_config(cfg,root/'config.json',prompts,plans_dir=root/'plans')
            self.assertEqual(len(resolved['protocol_prompt_hashes']),3)
            self.assertIn('Score 5',prompts.safety_judge)

if __name__=='__main__':unittest.main()
