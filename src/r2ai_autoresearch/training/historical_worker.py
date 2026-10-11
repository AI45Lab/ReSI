"""Run migrated historical DeepSpeed SFT through the common worker contract."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from ..portable.contracts import save
from ..executor import sha256_path


def encode_assistant_target(tokenizer, messages, max_length):
    """Keep a boundary token trainable when BPE merges template whitespace."""
    if [m['role'] for m in messages] != ['user', 'assistant']:
        raise RuntimeError('messages must be exactly [user, assistant]')
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix):
        raise RuntimeError('Supervised chat template does not preserve the generation prefix')
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if len(ids) > max_length:
        raise RuntimeError(f'fail-closed overlength {len(ids)} > {max_length}')
    boundary = 0
    for left, right in zip(prefix_ids, ids):
        if left != right:
            break
        boundary += 1
    if not boundary or boundary >= len(ids):
        raise RuntimeError('invalid assistant-only label mask')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * boundary + ids[boundary:], 'length': len(ids)}


def prepare_rows(rows, method, family):
    from .ministral_sft_encoding import critic_envelope, split_target
    prepared=[]
    for r in rows:
        if not r.get('target'):raise ValueError('Historical SFT requires constructed targets')
        row={'messages':[{'role':'user','content':r['prompt']},{'role':'assistant','content':r['target']}]}
        if family in ('ministral3', 'mistral3'):
            reasoning='';final=r['target'];task='general' if r['kind']=='utility' else 'response_safety'
            if method=='critic_sft' and r['kind']!='utility':
                label='unsafe' if final.endswith('Therefore, the response is unsafe.') else 'safe'
                reasoning,final=split_target(critic_envelope(final,label));task='critic_safety'
            row.update(task_type=task,user_text=r['prompt'],assistant_reasoning=reasoning,assistant_final=final)
        prepared.append(row)
    return prepared


def distributed_batch(global_batch, microbatch, world_size):
    divisor=microbatch*world_size
    if divisor<1 or global_batch%divisor:raise ValueError('training.batch_size must divide evenly across GPUs and SFT microbatches')
    return global_batch//divisor


def run(request):
    if request['operation']!='sft':raise ValueError('Historical SFT worker received another operation')
    cfg=request['config'];training=cfg['research_training'];execution=cfg['model_execution']
    if execution['dtype']!='bfloat16':raise ValueError('Historical DeepSpeed SFT requires model.dtype=bfloat16')
    if training['lora']['target_modules']!='model_default':raise ValueError('Historical SFT requires training.lora.target_modules=model_default (native text-model projections)')
    parent=Path(request['parent']).resolve();root=Path(request['output']);root.mkdir(parents=True,exist_ok=True)
    family=json.loads((parent/'config.json').read_text()).get('model_type')
    if family not in ('mistral3','qwen3_5','qwen3_5_text'):
        raise ValueError('Historical SFT currently covers recorded Qwen3.5/3.6 and Ministral3; use portable for small Qwen3')
    start=time.monotonic();base=parent
    if (parent/'r2ai_checkpoint.json').exists():
        from ..research.export import export_checkpoint
        base=Path(export_checkpoint(parent,Path(request['service_cache']).parent/'materialized_models'/sha256_path(parent),execution))
    import torch
    import pandas as pd
    world=torch.cuda.device_count()
    if world<1:raise RuntimeError('CUDA GPUs required')
    micro=training['sft']['microbatch'];accum=distributed_batch(cfg['batch_size'],micro,world)
    rows=prepare_rows(request['rows'],request['recipe']['method'],family)
    data=root/'train.parquet';pd.DataFrame(rows).to_parquet(data,index=False)
    text=root/'train.jsonl';text.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'state':'frozen_train_ready','formal_test_data_used':False,'system_role_present':False,
              'loss_mask':'assistant_target_only','max_sequence_length':cfg['max_sequence_length'],
              'artifacts':{'train.parquet':sha256_path(data),'train.jsonl':sha256_path(text)}}
    if family=='mistral3':manifest.update(model_family='ministral3',reasoning_protocol='mistral_square_think_v1')
    save(root/'data_manifest.json',manifest)
    ds={'bf16':{'enabled':True},'zero_optimization':{'stage':training['sft']['deepspeed_stage']},
        'train_micro_batch_size_per_gpu':micro,'gradient_accumulation_steps':accum,'train_batch_size':cfg['batch_size']}
    save(root/'deepspeed.json',ds)
    module='deepspeed_mixed_sft_ministral_official' if family=='mistral3' else 'deepspeed_critic_sft_a3'
    args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(world),
          '-m','r2ai_autoresearch.training.'+module,'--base',str(base),'--data',str(data),
          '--data-manifest',str(root/'data_manifest.json'),'--output',str(root/'trained'),
          '--deepspeed',str(root/'deepspeed.json'),'--microbatch',str(micro),'--grad-accum',str(accum),
          '--max-length',str(cfg['max_sequence_length']),'--max-steps',str(request['recipe']['steps']),
          '--learning-rate',str(request['recipe']['learning_rate']),'--lora-rank',str(training['lora']['rank']),
          '--lora-alpha',str(training['lora']['alpha']),'--lora-dropout',str(training['lora']['dropout']),
          '--seed',str(cfg['seed'])]
    if family=='mistral3':args.append('--lazy-tokenize')
    save(root/'launch.json',{'argv':args,'engine':'historical_deepspeed','global_batch':cfg['batch_size']})
    subprocess.run(args,check=True)
    adapter=root/'trained/adapter';status=root/'reload.json'
    subprocess.run([sys.executable,'-m','r2ai_autoresearch.training.verify_peft_reload_multifamily',
                    '--base',str(base),'--adapter',str(adapter),'--status',str(status),
                    '--model-family','ministral3' if family=='mistral3' else 'qwen36'],check=True)
    checkpoint=root/'checkpoint';checkpoint.mkdir(exist_ok=True);shutil.copytree(adapter,checkpoint/'adapter',dirs_exist_ok=True)
    from transformers import AutoTokenizer, GenerationConfig
    from ..research.model_adapter import copy_model_assets, describe
    AutoTokenizer.from_pretrained(base,local_files_only=True).save_pretrained(checkpoint)
    shutil.copy2(base/'config.json',checkpoint/'config.json')
    GenerationConfig.from_pretrained(base,local_files_only=True).save_pretrained(checkpoint);copy_model_assets(base,checkpoint)
    save(checkpoint/'r2ai_checkpoint.json',{'format':'r2ai.lora_checkpoint.v1','parent':str(base),
         'parent_sha256':sha256_path(base),'training_parent':str(parent),'adapter':'adapter',
         'model_adapter':describe(base,execution),'engine':'historical_deepspeed'})
    record=json.loads((root/'trained/training_run_manifest.json').read_text())
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_path(checkpoint),
            'optimizer_steps':record['steps'],'training_record':record,
            'reload':json.loads(status.read_text()),'seconds':time.monotonic()-start,'engine':'historical_deepspeed'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--result',required=True)
    a=p.parse_args();save(a.result,run(json.loads(Path(a.request).read_text())))

if __name__=='__main__':main()
