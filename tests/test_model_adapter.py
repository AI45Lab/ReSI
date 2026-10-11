import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from r2ai_autoresearch.research.model_adapter import describe,validate_checkpoint,encode_sft,copy_model_assets
from r2ai_autoresearch.research.controller import Controller
from test_research_protocol import Fake,cfg

class AdapterTests(unittest.TestCase):
    def test_capability_filter_skips_grpo_without_changing_selection(self):
        c=cfg();c['skills']['S4']['enabled']=True;c['search']['methods']=['critic_sft','direct_grpo']
        b=Fake(c);b.capabilities={'supported_methods':['critic_sft','a3'],'grpo_supported':False}
        r=Controller(c,b).run()
        self.assertEqual(r['skipped_methods'],['direct_grpo'])
        self.assertEqual(len(r['history'][0]['eligible']),2)
        self.assertFalse(any('/S4/' in x[0] or x[1]['method']=='direct_grpo' for x in b.calls))
        self.assertNotEqual(r['retained_model'],'M0')

    def test_no_supported_methods_rejected_before_execution(self):
        c=cfg();c['search']['methods']=['direct_grpo'];b=Fake(c)
        b.capabilities={'supported_methods':['critic_sft','a3'],'grpo_supported':False}
        with self.assertRaisesRegex(ValueError,'none'):Controller(c,b)
        self.assertEqual(b.calls,[])

    def test_official_serialization_and_custom_code_consent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);config=p/'config.json';settings={'backend':'single_gpu','trust_remote_code':False}
            config.write_text(json.dumps({'model_type':'mistral3'}))
            self.assertEqual(describe(p,settings)['serialization'],'mistral_common')
            config.write_text(json.dumps({'model_type':'safin1_35b_a3b','auto_map':{'AutoModelForCausalLM':'local.Model'}}))
            with self.assertRaisesRegex(ValueError,'trust_remote_code'):describe(p,settings)
            info=describe(p,{**settings,'trust_remote_code':True})
            self.assertFalse(info['grpo_supported']);self.assertEqual(info['supported_methods'],['critic_sft','a3'])

    def test_shard_coverage_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'config.json').write_text('{}')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'part-1.safetensors','b':'part-2.safetensors'}}))
            (p/'part-1.safetensors').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError,'part-2'):validate_checkpoint(p)
            (p/'part-2.safetensors').write_bytes(b'fixture');self.assertEqual(validate_checkpoint(p),p.resolve())

    def test_adapter_ancestry_and_cycle_check(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'config.json').write_text('{}');(p/'adapter').mkdir()
            (p/'adapter/adapter_config.json').write_text('{}');(p/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            (p/'r2ai_checkpoint.json').write_text(json.dumps({'format':'r2ai.lora_checkpoint.v1','parent':'.'}))
            with self.assertRaisesRegex(ValueError,'Cyclic'):validate_checkpoint(p)
            base=p/'base';base.mkdir();(base/'config.json').write_text('{}');(base/'model.safetensors').write_bytes(b'fixture')
            (p/'r2ai_checkpoint.json').write_text(json.dumps({'format':'r2ai.lora_checkpoint.v1','parent':'base'}));validate_checkpoint(p)

    def test_ministral_critic_goes_to_reasoning_channel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'config.json').write_text('{"model_type":"mistral3"}')
            row={'id':'x','kind':'harmful','prompt':'request','target':'It enables harm. Therefore, the response is unsafe.'}
            with patch('r2ai_autoresearch.training.ministral_sft_encoding.MinistralSFTEncoder') as cls:
                cls.return_value.encode.return_value={'input_ids':[1,2],'labels':[-100,2]}
                result=encode_sft(row,None,None,p,'critic_sft',128)
                cls.return_value.encode.assert_called_once_with('request','It enables harm.','Therefore, the response is unsafe.',allow_empty_reasoning=True)
                self.assertEqual(result,('x',[1,2],[-100,2]))

    def test_preserve_native_model_assets(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'base';dest=Path(d)/'trained';parent.mkdir();dest.mkdir()
            for name in ('tekken.json','SYSTEM_PROMPT.txt','modeling_local.py'):(parent/name).write_text(name)
            copy_model_assets(parent,dest)
            self.assertEqual(sorted(p.name for p in dest.iterdir()),['SYSTEM_PROMPT.txt','modeling_local.py','tekken.json'])
