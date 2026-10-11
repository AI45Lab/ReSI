"""Execute the migrated VERL/DAPO runner behind the shared trial interface."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from ..portable.contracts import save
from ..executor import sha256_path


def prepare_prompts(rows,system_prompt=None):
    result=[]
    for index,row in enumerate(rows):
        if row['kind'] not in ('harmful','benign'):raise ValueError('Safety GRPO accepts harmful/benign prompts, not general-task replay')
        result.append({'data_source':'r2ai_safety','prompt':([{'role':'system','content':system_prompt}] if system_prompt else [])+[{'role':'user','content':row['prompt']}],
                       'ability':'safety','reward_model':{'style':'rule','ground_truth':''},
                       'extra_info':{'index':index,'record_id':row['id'],'prompt_text':row['prompt'],'prompt_type':row['kind']}})
    if not result:raise ValueError('No prompts remain after GRPO screening')
    return result


def launch_environment(request,base,root,world,targets):
    cfg=request['config'];t=cfg['research_training'];g=t['grpo'];recipe=request['recipe']
    if world<1 or world%g['tensor_parallel_size']:raise ValueError('GRPO tensor parallel size must divide the visible GPU count')
    # The shared batch size counts sampled completions; VERL counts prompt groups.
    groups=cfg['batch_size']//g['num_generations']
    if groups<1:raise ValueError('GRPO batch has no complete rollout groups')
    return {key:str(value) for key,value in {
        'BASE':base,'RUN':root/'trained','DATA':root/'prompts.parquet','MODEL_FAMILY':targets[0],
        'R2AI_VERL_ROOT':g['runtime_path'],'TRAIN_BATCH_SIZE':groups,'PPO_MINI_BATCH_SIZE':groups,
        'TRAINER_N_GPUS_PER_NODE':world,'TRAINER_NNODES':1,'ROLLOUT_TP_SIZE':g['tensor_parallel_size'],
        'ROLLOUT_N':g['num_generations'],'ACTOR_LR':recipe['learning_rate'],'TOTAL_STEPS':recipe['steps'],
        'TOTAL_EPOCHS':max(100,recipe['steps']),'MAX_PROMPT_LENGTH':cfg['max_sequence_length'],
        'MAX_RESPONSE_LENGTH':cfg['max_new_tokens'],'LORA_RANK':t['lora']['rank'],'LORA_ALPHA':t['lora']['alpha'],
        'LORA_TARGETS':('"'+targets[1]+'"') if isinstance(targets[1],str) else json.dumps(targets[1]),'TEMPERATURE':g['temperature'],'TOP_P':g['top_p'],
        'TOP_K':g['top_k'] if g['top_k'] else -1,'CLIP_LOW':g['clip_low'],'CLIP_HIGH':g['clip_high'],
        'ROLLOUT_GPU_MEMORY_UTILIZATION':g['gpu_memory_utilization'],'MAX_NUM_GEN_BATCHES':g['max_generation_batches'],
        'R2AI_SERVICE_CACHE':request['service_cache'],'R2AI_MAX_API_CALLS':cfg['max_api_calls'],'SEED':cfg['seed'],
        'ENABLE_THINKING':str(cfg.get('model_execution',{}).get('enable_thinking',True)).lower(),
        'ADAPTER':'null','RESUME_MODE':'disable','SAVE_FREQ':-1,'SAVE_STEPS':'','EXPERIMENT_NAME':root.name,
    }.items()}


def normalize_qwen_adapter(adapter, output):
    """Convert VERL's conditional-generation text prefix to the recorded causal loader.

    Keep the original VERL export intact. Strict tensor reload follows this
    namespace-only conversion; architecture/shape differences cannot pass it.
    """
    from safetensors.torch import load_file,save_file
    weights=load_file(adapter/'adapter_model.safetensors')
    if not any('.model.language_model.' in key for key in weights):return adapter
    renamed={key.replace('.model.language_model.','.model.'):value for key,value in weights.items()}
    if len(renamed)!=len(weights):raise ValueError('Adapter namespace conversion caused duplicate keys')
    if any('.visual.' in key for key in renamed):raise ValueError('Unexpected vision adapter in text-only safety training')
    shutil.copytree(adapter,output)
    save_file(renamed,output/'adapter_model.safetensors')
    save(output/'namespace_conversion.json',{'operation':'Qwen conditional text prefix to causal prefix','tensors':len(renamed)})
    return output


def validate_text_adapter_keys(keys, pattern, expected_count):
    """Reject unsupported tensors before narrowing an exported PEFT target pattern."""
    if len(keys) != expected_count:
        raise ValueError('Unexpected tensor count in the text adapter')
    modules = {}
    for key in keys:
        match = re.fullmatch(r'base_model\.model\.(.+)\.lora_([AB])\.weight', key)
        if not match or not re.fullmatch(pattern, match[1]):
            raise ValueError('Adapter contains a tensor outside the declared text modules: '+key)
        modules.setdefault(match[1], set()).add(match[2])
    if any(sides != {'A', 'B'} for sides in modules.values()):
        raise ValueError('Text adapter has an incomplete LoRA A/B pair')


def normalize_text_adapter(adapter, output, family):
    """Preserve tensors while restoring the exact text-only PEFT target pattern.

    VERL's merger may export short projection names. On a multimodal loader
    these also match the vision tower, creating additional untrained adapters.
    Validate the saved tensors first; never discard unexpected tensors.
    """
    from safetensors import safe_open
    with safe_open(adapter/'adapter_model.safetensors', framework='pt', device='cpu') as f:
        validate_text_adapter_keys(list(f.keys()), family.lora_target, family.expected_adapter_tensors)
    config = json.loads((adapter/'adapter_config.json').read_text())
    if config.get('target_modules') == family.lora_target:
        return adapter
    shutil.copytree(adapter, output)
    config['target_modules'] = family.lora_target
    save(output/'adapter_config.json', config)
    save(output/'target_normalization.json', {
        'operation': 'Restore exact text-only module pattern from validated tensor keys',
        'tensor_file_unchanged': True,
        'adapter_sha256': sha256_path(adapter/'adapter_model.safetensors'),
        'target_modules': family.lora_target})
    return output


def run(request):
    if request['operation']!='grpo':raise ValueError('VERL worker requires a GRPO trial')
    cfg=request['config'];t=cfg['research_training'];execution=cfg['model_execution']
    if execution['dtype']!='bfloat16':raise ValueError('Historical VERL requires model.dtype=bfloat16')
    if t['lora']['target_modules']!='model_default' or t['lora']['dropout']!=0:
        raise ValueError('Historical VERL requires model_default LoRA targets and zero dropout')
    runtime=Path(t['grpo']['runtime_path']).resolve();python=runtime/'.venv/bin/python'
    if not python.is_file():raise ValueError('Install the pinned runtime with scripts/setup_verl.sh')
    parent=Path(request['parent']).resolve();root=Path(request['output']).resolve();root.mkdir(parents=True,exist_ok=True)
    from .model_family import resolve_model_family
    family=resolve_model_family(parent)
    start=time.monotonic();base=parent
    if (parent/'r2ai_checkpoint.json').exists():
        from ..research.export import export_checkpoint
        base=Path(export_checkpoint(parent,Path(request['service_cache']).parent/'materialized_models'/sha256_path(parent),execution))
    import pandas as pd
    import torch
    from ..portable.gpu import grpo_prompt_batch
    sampled,batch_info=grpo_prompt_batch(request['rows'],cfg['batch_size'],t['grpo']['num_generations'])
    system_prompt=(base/'SYSTEM_PROMPT.txt').read_text().strip() if family.name=='ministral3' else cfg.get('evaluation',{}).get('system_prompt','')
    rows=prepare_prompts(sampled,system_prompt)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    for row in rows:
        ids=tokenizer.apply_chat_template(row['prompt'],tokenize=True,return_dict=False,add_generation_prompt=True,
                                          enable_thinking=execution.get('enable_thinking',True))
        prefix=tokenizer.decode(ids,skip_special_tokens=False).rstrip()
        row['extra_info']['reasoning_started']=prefix.endswith(('<think>','[THINK]'))
    pd.DataFrame(rows).to_parquet(root/'prompts.parquet',index=False)
    save(root/'prompt_batch.json',batch_info)
    targets=family.lora_target
    if family.name=='qwen36':targets=targets.replace(r'^model\.layers',r'^model(?:\.language_model)?\.layers')
    env=launch_environment(request,base,root,torch.cuda.device_count(),(family.name,targets))
    save(root/'launch.json',{'engine':'historical_verl','settings':env,'prompt_rows':len(rows),
                           'screening_responses_reused':False,'parent_sha256':sha256_path(base)})
    script=Path(__file__).resolve().parents[3]/'scripts/run_historical_verl.sh'
    subprocess.run(['bash',str(script)],env={**os.environ,**env},check=True)
    return finalize_checkpoint(request, base, root, family, start)


def finalize_checkpoint(request, base, root, family, start):
    """Export and strictly reload a completed native training checkpoint."""
    import torch
    execution=request['config']['model_execution']
    python=Path(request['config']['research_training']['grpo']['runtime_path'])/'.venv/bin/python'
    parent=Path(request['parent']).resolve()
    actor=root/'trained/checkpoints'/('global_step_'+str(request['recipe']['steps']))/'actor'
    adapter=actor/'lora_adapter'
    if not (adapter/'adapter_model.safetensors').is_file():
        if not actor.is_dir():raise RuntimeError('VERL did not complete the requested number of optimizer steps')
        merged=root/'adapter_export'
        if not (merged/'lora_adapter/adapter_model.safetensors').is_file():
            subprocess.run([str(python),'-m','verl.model_merger','merge','--backend','fsdp',
                            '--local_dir',str(actor),'--target_dir',str(merged)],check=True)
        adapter=merged/'lora_adapter'
    if not (adapter/'adapter_model.safetensors').is_file() or not (adapter/'adapter_config.json').is_file():
        raise RuntimeError('VERL completed without an exportable adapter')
    if family.name=='qwen36':adapter=normalize_qwen_adapter(adapter,root/'causal_adapter')
    adapter=normalize_text_adapter(adapter,root/'text_adapter',family)
    from safetensors.torch import load_file
    weights=load_file(adapter/'adapter_model.safetensors')
    if any(not torch.isfinite(v).all().item() for v in weights.values()):raise RuntimeError('Nonfinite GRPO adapter')
    b_norm=sum(v.float().square().sum().item() for k,v in weights.items() if '.lora_B.' in k)**.5
    if b_norm<=0:raise RuntimeError('No measured GRPO adapter update')
    status=root/'reload.json'
    subprocess.run([sys.executable,'-m','r2ai_autoresearch.training.verify_peft_reload_multifamily',
        '--base',str(base),'--adapter',str(adapter),'--status',str(status),'--model-family',family.name],check=True)
    checkpoint=root/'checkpoint';checkpoint.mkdir(exist_ok=True);shutil.copytree(adapter,checkpoint/'adapter',dirs_exist_ok=True)
    from transformers import AutoTokenizer
    from ..research.model_adapter import copy_model_assets,describe
    AutoTokenizer.from_pretrained(base,local_files_only=True).save_pretrained(checkpoint)
    shutil.copy2(base/'config.json',checkpoint/'config.json')
    if (base/'generation_config.json').is_file():shutil.copy2(base/'generation_config.json',checkpoint/'generation_config.json')
    copy_model_assets(base,checkpoint)
    save(checkpoint/'r2ai_checkpoint.json',{'format':'r2ai.lora_checkpoint.v1','parent':str(base),
         'parent_sha256':sha256_path(base),'training_parent':str(parent),'adapter':'adapter',
         'model_adapter':describe(base,execution),'engine':'historical_verl'})
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_path(checkpoint),
            'optimizer_steps':request['recipe']['steps'],'lora_b_norm':b_norm,'reload':json.loads(status.read_text()),
            'seconds':time.monotonic()-start,'engine':'historical_verl'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--result',required=True)
    a=p.parse_args();save(a.result,run(json.loads(Path(a.request).read_text())))

if __name__=='__main__':main()
