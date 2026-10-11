"""Model serialization and execution capabilities, independent of search policy.

The controller consumes supported methods, never model names. Model checkpoints
supply tokenizer/templates; Ministral critic targets use its official encoder.
"""
from pathlib import Path
import json
import shutil

METHODS = ('critic_sft', 'a3', 'direct_grpo')


def describe(path, settings):
    c = json.loads((Path(path) / 'config.json').read_text())
    family = c.get('model_type', '')
    backend = settings['backend']
    if backend not in ('single_gpu', 'hf_sharded'):
        raise ValueError('Unknown model backend: ' + backend)
    custom = bool(c.get('auto_map')) and family.startswith('safin')
    if custom and not settings.get('trust_remote_code', False):
        raise ValueError('This model requires explicitly setting model.trust_remote_code=true')
    methods = list(METHODS)
    if family.startswith('safin'):
        methods.remove('direct_grpo')
    return {'model_type': family, 'backend': backend, 'supported_methods': methods,
            'grpo_supported': 'direct_grpo' in methods,
            'serialization': 'mistral_common' if family == 'mistral3' else 'checkpoint_chat_template',
            'loader': 'image_text' if family in ('mistral3', 'qwen3_5_moe') else 'causal',
            'grpo_unavailable_reason': 'Safin GRPO backend is not validated' if family.startswith('safin') else None}


def validate_checkpoint(path, _ancestors=()):
    """Check weight coverage and adapter ancestry before expensive GPU work."""
    path=Path(path).resolve()
    if path in _ancestors:raise ValueError('Cyclic checkpoint ancestry')
    if not (path/'config.json').is_file():raise ValueError('Missing model config.json')
    manifest=path/'r2ai_checkpoint.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta.get('format')!='r2ai.lora_checkpoint.v1':raise ValueError('Unknown checkpoint format')
        if not (path/'adapter/adapter_config.json').is_file() or not (path/'adapter/adapter_model.safetensors').is_file():
            raise ValueError('Incomplete adapter checkpoint')
        validate_checkpoint(path/meta['parent'],(*_ancestors,path))
    else:
        index=path/'model.safetensors.index.json'
        names=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        missing=sorted(name for name in names if not (path/name).is_file())
        if missing:raise ValueError('Missing model weight files: '+', '.join(missing))
    return path


def load_model(path, settings, _ancestors=()):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    path=validate_checkpoint(path)
    if path in _ancestors:raise ValueError('Cyclic checkpoint ancestry')
    info = describe(path, settings)
    manifest=path/'r2ai_checkpoint.json'
    if manifest.exists():
        from peft import PeftModel
        meta=json.loads(manifest.read_text())
        parent=(path/meta['parent']).resolve()
        if meta.get('parent_sha256'):
            from ..executor import sha256_path
            if sha256_path(parent)!=meta['parent_sha256']:raise ValueError('Checkpoint parent content changed')
        model,_=load_model(parent,settings,(*_ancestors,path))
        if isinstance(model,PeftModel):model=model.merge_and_unload(safe_merge=True)
        return PeftModel.from_pretrained(model,path/'adapter',is_trainable=False),info
    cls = AutoModelForImageTextToText if info['loader'] == 'image_text' else AutoModelForCausalLM
    kwargs = {'dtype': getattr(torch, settings.get('dtype', 'float32')),
              'local_files_only': True, 'trust_remote_code': settings.get('trust_remote_code', False),
              'attn_implementation': 'sdpa', 'device_map': {'': 0}}
    if settings['backend'] == 'hf_sharded':
        kwargs['device_map'] = 'balanced'
        # Keep activation headroom on every GPU; no hidden CPU/disk offloading.
        kwargs['max_memory'] = {i: int(torch.cuda.get_device_properties(i).total_memory * .80)
                                for i in range(torch.cuda.device_count())}
        kwargs['max_memory']['cpu'] = 0
    model = cls.from_pretrained(path, **kwargs)
    if any(str(d) in ('cpu', 'disk') for d in getattr(model, 'hf_device_map', {}).values()):
        raise RuntimeError('Model exceeds GPU capacity; choose more GPUs or a distributed backend')
    return model, info


def copy_model_assets(parent, checkpoint):
    """Preserve official tokenizer assets and custom architecture code on reload."""
    parent, checkpoint = Path(parent), Path(checkpoint)
    for pattern in ('*.py', 'tekken.json', 'SYSTEM_PROMPT.txt'):
        for path in parent.glob(pattern):
            shutil.copy2(path, checkpoint / path.name)


def encode_sft(row, tokenizer, prefix, parent, method, max_length):
    c = json.loads((Path(parent) / 'config.json').read_text())
    if c.get('model_type') == 'mistral3':
        from ..training.ministral_sft_encoding import MinistralSFTEncoder, split_target, critic_envelope
        encoder = MinistralSFTEncoder(parent, max_length)
        if method == 'critic_sft' and row.get('kind') != 'utility':
            label = 'unsafe' if row['target'].endswith('Therefore, the response is unsafe.') else 'safe'
            reasoning, final = split_target(critic_envelope(row['target'], label))
        else:
            target = row['target']
            if target.startswith('[THINK]') and '[/THINK]' in target:
                reasoning, final = target[7:].split('[/THINK]', 1)
            else:
                reasoning, final = '', target
        encoded = encoder.encode(row['prompt'], reasoning, final, allow_empty_reasoning=True)
        return row['id'], encoded['input_ids'], encoded['labels']
    pre = tokenizer.encode(prefix(row), add_special_tokens=False)
    target = tokenizer.encode(row['target'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(pre) + len(target) > max_length:
        raise ValueError('Target exceeds max_sequence_length; supervision is never silently truncated')
    return row['id'], pre + target, [-100] * len(pre) + target


def encode_native_generation(tokenizer, conversations):
    """Preserve native control tokens; never encode a decoded chat template."""
    return tokenizer.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
                                         padding=True,return_tensors='pt',return_dict=True)
