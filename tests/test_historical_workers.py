import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from r2ai_autoresearch.research.config import validate
from r2ai_autoresearch.training.historical_worker import distributed_batch,prepare_rows,encode_assistant_target
from r2ai_autoresearch.training.verl_worker import prepare_prompts,launch_environment,validate_text_adapter_keys

class HistoricalWorkerTests(unittest.TestCase):
    def test_exported_text_adapter_cannot_silently_add_vision_modules(self):
        pattern=r'^model\.language_model\.layers\.\d+\.self_attn\.q_proj$'
        keys=['base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_'+side+'.weight' for side in ('A','B')]
        validate_text_adapter_keys(keys,pattern,2)
        with self.assertRaisesRegex(ValueError,'outside the declared text modules'):
            validate_text_adapter_keys([k.replace('language_model','vision_tower') for k in keys],pattern,2)
        with self.assertRaisesRegex(ValueError,'tensor count'):
            validate_text_adapter_keys(keys[:1],pattern,2)
        with self.assertRaisesRegex(ValueError,'incomplete LoRA'):
            validate_text_adapter_keys([keys[0],keys[1].replace('layers.0','layers.1')],pattern,2)

    def test_verl_rewards_exclude_native_reasoning(self):
        from r2ai_autoresearch.training.verl_reward import final_reward_text
        self.assertEqual(final_reward_text('[THINK]private reasoning[/THINK][RESPONSE]Final answer</s>',{},['</s>']),'Final answer')
        self.assertEqual(final_reward_text('reasoning</think>Final<|im_end|>',{'reasoning_started':True},['<|im_end|>']),'Final')
        with self.assertRaisesRegex(ValueError,'Incomplete reasoning'):
            final_reward_text('unfinished reasoning</s>',{'reasoning_started':True},['</s>'])

    def test_native_generation_keeps_tokenized_chat_template(self):
        from r2ai_autoresearch.research.model_adapter import encode_native_generation
        class NativeTokenizer:
            def apply_chat_template(self,conversations,**kwargs):
                assert kwargs['tokenize'] is True
                assert kwargs['return_dict'] is True
                return {'input_ids': [[1, 42, 3]], 'attention_mask': [[1, 1, 1]]}
            def __call__(self,*args,**kwargs):raise AssertionError('Do not retokenize native templates')
        self.assertEqual(encode_native_generation(NativeTokenizer(),[[{'role':'user','content':'hello'}]])['input_ids'],[[1,42,3]])

    def test_ministral_completed_answer_does_not_require_prefix_ids(self):
        from r2ai_autoresearch.training.ministral_sft_encoding import MinistralSFTEncoder
        encoder=MinistralSFTEncoder.__new__(MinistralSFTEncoder)
        encoder.max_length=100;encoder.system_prompt='System'
        for name in ('TextChunk','ThinkChunk','AssistantMessage','SystemMessage','UserMessage','ChatCompletionRequest'):
            setattr(encoder,name,lambda **kw:SimpleNamespace(**kw))
        encoder.mistral=SimpleNamespace(
            encode_chat_completion=lambda request:SimpleNamespace(tokens=[1,2,34,8,35,9,2],prefix_ids=None),
            instruct_tokenizer=SimpleNamespace(encode_assistant_message=lambda message,**kw:[34,8,35,9,2]))
        encoded=encoder.encode('hello','Reasoning','Hello!')
        self.assertEqual(encoded['labels'],[-100,-100,34,8,35,9,2])
        encoder.mistral.instruct_tokenizer.encode_assistant_message=lambda *a,**kw:[999]
        with self.assertRaisesRegex(RuntimeError,'suffix'):
            encoder.encode('hello','Reasoning','Hello!')

    def test_template_boundary_merge_remains_supervised(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return 'user|assistant\n' if len(messages)==1 else 'user|assistant\n\nanswer'
            def __call__(self, text, **kwargs):
                return {'input_ids': [10,20,30] if text.endswith('assistant\n') else [10,20,31,40]}
        messages=[{'role':'user','content':'hello'},{'role':'assistant','content':'answer'}]
        encoded=encode_assistant_target(Tokenizer(),messages,10)
        self.assertEqual(encoded['labels'],[-100,-100,31,40])
        with self.assertRaisesRegex(RuntimeError,'overlength'):
            encode_assistant_target(Tokenizer(),messages,3)
        class ChangedTemplate(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return 'user|assistant\n' if len(messages)==1 else 'different prompt|answer'
        with self.assertRaisesRegex(RuntimeError,'generation prefix'):
            encode_assistant_target(ChangedTemplate(),messages,10)

    def test_batch_units_are_explicit_and_not_silently_rounded(self):
        self.assertEqual(distributed_batch(32,1,4),8)
        with self.assertRaises(ValueError):distributed_batch(7,1,4)
        c=validate(json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text()))
        cfg={'research_training':c['training'],'batch_size':32,'max_sequence_length':4096,'max_new_tokens':2048,'max_api_calls':1000,'seed':42}
        request={'config':cfg,'recipe':{'steps':62,'learning_rate':1e-6},'service_cache':'cache'}
        env=launch_environment(request,Path('base'),Path('run'),4,('qwen36','regex'))
        self.assertEqual(env['TRAIN_BATCH_SIZE'],'4');self.assertEqual(env['ROLLOUT_N'],'8')
        self.assertEqual(env['TOTAL_STEPS'],'62');self.assertEqual(env['TRAINER_N_GPUS_PER_NODE'],'4')
        for enabled in (False,True):
            cfg['model_execution']={'enable_thinking':enabled}
            env=launch_environment(request,Path('base'),Path('run'),4,('qwen36','regex'))
            self.assertEqual(env['ENABLE_THINKING'],str(enabled).lower())
        pattern=r'^model\.layers\.\d+\.q_proj$'
        env=launch_environment(request,Path('base'),Path('run'),4,('qwen36',pattern))
        self.assertEqual(env['LORA_TARGETS'],'"'+pattern+'"')
        c['training']['grpo']['tensor_parallel_size']=3
        with self.assertRaises(ValueError):launch_environment(request,Path('base'),Path('run'),4,('qwen36','regex'))

    def test_grpo_prompt_manifest_has_reward_metadata_and_no_cached_completions(self):
        row=prepare_prompts([{'id':'x','kind':'benign','prompt':'hello','target':'old answer'}])[0]
        self.assertEqual(row['extra_info']['prompt_type'],'benign')
        self.assertEqual(row['prompt'],[{'role':'user','content':'hello'}])
        self.assertNotIn('old answer',json.dumps(row))
        native=prepare_prompts([{'id':'x','kind':'benign','prompt':'hello'}],'native system')[0]
        self.assertEqual(native['prompt'][0],{'role':'system','content':'native system'})
        with self.assertRaises(ValueError):prepare_prompts([{'id':'x','kind':'utility','prompt':'hello'}])

    def test_ministral_critic_and_general_channels(self):
        rows=prepare_rows([{'prompt':'judge this','target':'Reason. Therefore, the response is unsafe.','kind':'harmful'},
                           {'prompt':'hello','target':'Hello!','kind':'utility'}],'critic_sft','ministral3')
        self.assertEqual(rows[0]['task_type'],'critic_safety')
        self.assertEqual(rows[0]['assistant_final'],'Therefore, the response is unsafe.')
        self.assertEqual(rows[1]['assistant_reasoning'],'');self.assertEqual(rows[1]['assistant_final'],'Hello!')
        raw_config_rows=prepare_rows([{'prompt':'judge this','target':'Reason. Therefore, the response is unsafe.','kind':'harmful'},
                                     {'prompt':'hello','target':'Hello!','kind':'utility'}],'critic_sft','mistral3')
        self.assertEqual(rows,raw_config_rows)

if __name__=='__main__':unittest.main()
