"""Single-GPU training/generation worker. Imports heavy libraries only on launch."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

from .contracts import save
from .evaluate import judge
from .service import Client
from ..executor import sha256_path


SYSTEM = "Be helpful and safe. Keep your answer concise, at most 60 words, and follow the user's requested format."


def messages(row, system=SYSTEM):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": row["prompt"]}]


def require_update(operation, delta, grad_norms, root, reward_groups, clipping=None):
    valid = math.isfinite(delta) and delta > 0 and any(math.isfinite(g) and g > 0 for g in grad_norms)
    if valid: return
    if operation == "grpo":
        save(Path(root) / "signal_failure.json", {
            "issue": "no_usable_grpo_update",
            "weight_delta_l2": delta if math.isfinite(delta) else None,
            "gradient_norms": [g if math.isfinite(g) else None for g in grad_norms],
            "groups": reward_groups, "completion_clipped_ratios": clipping or [],
            "note": "Reward variance alone does not imply an unmasked policy gradient. No valid checkpoint exported."})
    raise ValueError("no measured optimizer update")


def grpo_prompt_batch(rows,batch_size,num_generations):
    """Cycle a tiny screened pool so TRL's group sampler can form one batch."""
    if not rows:raise ValueError('No GRPO prompts after screening')
    minimum=batch_size//num_generations
    if batch_size%num_generations or minimum<1:raise ValueError('Invalid GRPO batch/group sizes')
    expanded=list(rows)
    while len(expanded)<minimum:expanded.append(rows[len(expanded)%len(rows)])
    return expanded,{'input_rows':len(rows),'unique_prompt_ids':len({r['id'] for r in rows}),
                     'sampler_rows':len(expanded),'cycled_rows':len(expanded)-len(rows)}


def checkpoint_steps(total_steps, maximum):
    """Bounded, evenly spaced optimizer steps, always including the last step."""
    if type(total_steps) is not int or total_steps < 1 or type(maximum) is not int or maximum < 1:
        raise ValueError('Positive checkpoint step budget required')
    count=min(total_steps,maximum)
    return sorted({math.ceil(total_steps*i/count) for i in range(1,count+1)})


def run(request):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    cfg = request["config"]; random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    root = Path(request["output"]); root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(request["parent"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    execution=cfg.get('model_execution', {'backend':'single_gpu','dtype':'float32','trust_remote_code':False})
    generation_dtype = cfg.get("evaluation", {}).get("dtype", "float32")
    load_dtype = torch.bfloat16 if request["operation"] == "generate" and generation_dtype == "bfloat16" else torch.float32
    if load_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("registered bfloat16 evaluation requires a supported GPU")
    from ..research.model_adapter import load_model, encode_sft, copy_model_assets
    model, adapter_info = load_model(request['parent'], execution)
    input_device=model.get_input_embeddings().weight.device
    if request['operation'] in ('grpo','grpo_screen') and not adapter_info['grpo_supported']:
        raise ValueError('Backend does not support GRPO for this model')
    model.config.use_cache = True
    rows = request["rows"]
    system = cfg.get("evaluation", {}).get("system_prompt", SYSTEM)
    if adapter_info['serialization']=='mistral_common':
        native_system=(Path(request['parent'])/'SYSTEM_PROMPT.txt').read_text().strip()
        if system not in (None,"",native_system):
            raise ValueError('Ministral requires its native system prompt for consistent training/evaluation')
        system=native_system
    cap = cfg.get("evaluation", {}).get("max_new_tokens", cfg["max_new_tokens"]) if request["operation"] == "generate" else cfg["max_new_tokens"]
    def prefix(row):
        if adapter_info['serialization']=='mistral_common':
            ids=tokenizer.apply_chat_template(messages(row,system),tokenize=True,add_generation_prompt=True,return_dict=False)
            return tokenizer.decode(ids,skip_special_tokens=False)
        return tokenizer.apply_chat_template(messages(row, system), tokenize=False, add_generation_prompt=True, enable_thinking=cfg.get("enable_thinking", False))
    def encode_prompts(batch):
        if adapter_info['serialization']=='mistral_common':
            from ..research.model_adapter import encode_native_generation
            return encode_native_generation(tokenizer,[messages(row,system) for row in batch]).to(input_device)
        return tokenizer([prefix(row) for row in batch],return_tensors='pt',padding=True).to(input_device)
    from ..research.responses import decode_response
    original_generation = __import__('copy').deepcopy(model.generation_config)
    if request["operation"] == "grpo_screen":
        from ..research.reward import judge_rewards
        settings=cfg['research_training']['grpo'];screen=settings['screening']
        candidates=list({r['id']:r for r in rows}.values())
        if screen['max_prompts']:candidates=candidates[:screen['max_prompts']]
        client=Client(request['service_cache'],cfg['max_api_calls'])
        groups=[];selected=[];model.eval();width=cfg['num_generations']
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        for row in candidates:
            siblings=[]
            for offset in range(0,width,screen['batch_size']):
                size=min(screen['batch_size'],width-offset)
                encoded=encode_prompts([row]*size)
                with torch.inference_mode():
                    ids=model.generate(**encoded,do_sample=True,temperature=settings['temperature'],top_p=settings['top_p'],
                                       top_k=settings['top_k'],max_new_tokens=cfg['max_new_tokens'],pad_token_id=tokenizer.pad_token_id)
                for completion in ids[:,encoded.input_ids.shape[1]:]:
                    tokens=completion.tolist();decoded=decode_response(tokenizer,tokens,prefix(row))
                    siblings.append({**decoded,'complete':bool(eos.intersection(tokens)) and decoded['final_complete']})
            valid=all(x['complete'] and x['text'] for x in siblings)
            if valid:
                values,judgments=judge_rewards(client,[row]*width,[x['text'] for x in siblings],cfg.get('evaluation',{}).get('judge_concurrency',1))
            else:values=[];judgments=[]
            keep=valid and len(set(values))>1
            if keep:selected.append(row['id'])
            record={'id':row['id'],'rewards':values,'complete':valid,'selected':keep,'outputs':siblings,'judgments':judgments}
            groups.append({k:v for k,v in record.items() if k not in ('outputs','judgments')})
            with (root/'screening.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({'stage':'grpo_screen','screened':len(groups),'selected':len(selected)}),flush=True)
        return {'selected_ids':selected,'screened_prompts':len(groups),'groups':groups,
                'temperature':settings['temperature'],'top_p':settings['top_p'],'top_k':settings['top_k'],
                'num_generations':width,'seconds':time.monotonic()-start}
    if request["operation"] == "generate":
        from ..research.generation import resolve, completion_end
        defaults=model.generation_config.to_dict()
        overrides=cfg.get('evaluation',{}).get('generation')
        # Legacy callers retain their registered greedy protocol.
        generation=resolve(defaults,overrides if overrides is not None else {'do_sample':False})
        model.generation_config=transformers.GenerationConfig.from_dict(generation)
        model.eval(); outputs = []
        generation_batch = cfg.get("evaluation", {}).get("batch_size", cfg["batch_size"])
        for offset in range(0, len(rows), generation_batch):
            batch = rows[offset:offset + generation_batch]
            encoded = encode_prompts(batch)
            with torch.inference_mode():
                ids = model.generate(**encoded, max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id)
            for row, completion in zip(batch, ids[:, encoded.input_ids.shape[1]:]):
                tokens = completion.tolist()
                end=completion_end(tokens,model.generation_config.eos_token_id)
                decoded=decode_response(tokenizer,tokens,prefix(row))
                outputs.append({"id": row["id"], **decoded,
                                "hit_token_limit": end is None or not decoded['final_complete'],
                                "tokens": tokens[:end+1] if end is not None else tokens})
            print(json.dumps({"stage": "generate", "completed": len(outputs), "total": len(rows), "seconds": time.monotonic()-start}), flush=True)
        return {"outputs": outputs, "seconds": time.monotonic() - start,
                "generation_dtype": execution.get("dtype",generation_dtype), "generation_config":generation, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    recipe = request["recipe"]
    lora = cfg.get("research_training", {}).get("lora", {})
    if isinstance(model,PeftModel):model=model.merge_and_unload(safe_merge=True)
    if lora.get('target_modules')=='model_default':
        from ..training.model_family import resolve_model_family
        lora={**lora,'target_modules':resolve_model_family(request['parent']).lora_target}
    model = get_peft_model(model, LoraConfig(r=lora.get("rank", 8), lora_alpha=lora.get("alpha", 16), target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
                                           lora_dropout=lora.get("dropout", 0.), task_type="CAUSAL_LM"))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in parameters.items()}
    losses, grad_norms, exposure = [], [], []
    reward_groups = []
    selection = recipe.get('_checkpoint_selection', {})
    planned = checkpoint_steps(recipe['steps'], selection['max_checkpoints']) if selection.get('enabled') else []
    saved_candidates = []
    def save_checkpoint(destination, step):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination / 'adapter', safe_serialization=True)
        tokenizer.save_pretrained(destination)
        model.config.save_pretrained(destination)
        original_generation.save_pretrained(destination)
        copy_model_assets(request['parent'], destination)
        save(destination/'r2ai_checkpoint.json', {'format':'r2ai.lora_checkpoint.v1',
            'parent':str(Path(request['parent']).resolve()), 'parent_sha256':request.get('parent_sha256'),
            'adapter':'adapter', 'model_adapter':adapter_info, 'optimizer_step':step})
        return {'checkpoint':str(destination.resolve()), 'checkpoint_sha256':sha256_path(destination), 'step':step}
    def save_intermediate(step):
        if step in planned and step < recipe['steps']:
            saved_candidates.append(save_checkpoint(root/'saved_checkpoints'/f'step-{step}',step))
    if request["operation"] == "sft":
        encoded = []
        for row in rows:
            encoded.append(encode_sft(row,tokenizer,prefix,request['parent'],recipe.get('method','critic_sft'),cfg.get('max_sequence_length',1024)))
        from ..research.diagnostics import supervision_diagnostics
        encoding_audit=supervision_diagnostics(encoded,tokenizer.eos_token_id)
        save(root/'supervision_diagnostics.json',encoding_audit)
        if encoding_audit['zero_supervision_ids']:
            raise ValueError('SFT examples have no supervised tokens; inspect supervision_diagnostics.json')
        order = list(range(len(encoded))); random.shuffle(order)
        optimizer = torch.optim.AdamW(parameters.values(), lr=recipe["learning_rate"], weight_decay=0.)
        model.train()
        for step in range(recipe["steps"]):
            batch = [encoded[order[(step * cfg["batch_size"] + i) % len(order)]] for i in range(cfg["batch_size"])]
            length = max(len(x[1]) for x in batch)
            # Right padding with explicit mask for supervised training.
            ids = torch.tensor([x[1] + [tokenizer.pad_token_id] * (length - len(x[1])) for x in batch], device=input_device)
            labels = torch.tensor([x[2] + [-100] * (length - len(x[2])) for x in batch], device=input_device)
            mask = torch.tensor([[1] * len(x[1]) + [0] * (length - len(x[1])) for x in batch], device=input_device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("non-finite SFT loss")
            loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(), 1.))
            if not norm > 0 or not torch.isfinite(torch.tensor(norm)):
                raise ValueError("no finite nonzero SFT gradient")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach())); grad_norms.append(norm)
            save_intermediate(step + 1)
            exposure.extend({"id": x[0], "loss_tokens": sum(t != -100 for t in x[2])} for x in batch)
            print(json.dumps({"stage": "sft", "step": step + 1, "loss": losses[-1], "gradient_norm": norm}), flush=True)
    elif request["operation"] == "grpo":
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        client = Client(request["service_cache"], cfg["max_api_calls"])
        row_map = {r["id"]: r for r in rows}
        def reward(completions, row_id, **kwargs):
            texts = [x[0]["content"] if isinstance(x, list) else x for x in completions]
            batch = [row_map[i] for i in row_id]
            token_rows=kwargs.get('completion_ids')
            if token_rows is not None:
                if len(token_rows)!=len(batch):raise ValueError('Reward token coverage mismatch')
                texts=[decode_response(tokenizer,ids,prefix(row))['text'] for row,ids in zip(batch,token_rows)]
            elif adapter_info['serialization']=='mistral_common' or cfg.get('enable_thinking',False):
                raise RuntimeError('Reasoning GRPO requires completion_ids from the trainer; refusing ambiguous final-answer judging')
            if cfg.get("research_training", {}).get("grpo", {}).get("reward") == "historical_binary_v2":
                from ..research.reward import judge_rewards
                values, judged = judge_rewards(client, batch, texts, cfg.get("evaluation", {}).get("judge_concurrency", 1))
            else:
                judged = judge(client, batch, texts)
                values = [0. if j.get("unsafe", False) else float(j["quality"]) / 4 for j in judged]
            width = cfg["num_generations"]
            for i in range(0, len(values), width):
                group = values[i:i + width]
                reward_groups.append({"row_id": row_id[i], "rewards": group, "nonconstant": len(set(group)) > 1})
            with (root / "rollouts.jsonl").open("a") as f:
                for row, text, j, value in zip(batch, texts, judged, values):
                    f.write(json.dumps({"id": row["id"], "text": text, "judgment": j, "reward": value}) + "\n")
            return values
        sampler_rows,prompt_sampling=grpo_prompt_batch(rows,cfg["batch_size"],cfg["num_generations"])
        save(root/"prompt_sampling.json",prompt_sampling)
        dataset = Dataset.from_list([{"prompt": messages(row, system), "row_id": row["id"]} for row in sampler_rows])
        grpo_settings = cfg.get("research_training", {}).get("grpo", {})
        args = GRPOConfig(output_dir=str(root / "trainer"), max_steps=recipe["steps"],
                          per_device_train_batch_size=cfg["batch_size"], num_generations=cfg["num_generations"],
                          learning_rate=recipe["learning_rate"], gradient_accumulation_steps=1,
                          max_completion_length=cfg["max_new_tokens"], temperature=grpo_settings.get("temperature", 1.), top_p=grpo_settings.get("top_p", .95), top_k=grpo_settings.get("top_k", 0),
                          loss_type="dapo", epsilon=grpo_settings.get("clip_low", .2), epsilon_high=grpo_settings.get("clip_high", .28), beta=0., scale_rewards="group",
                          use_vllm=False, bf16=execution.get("dtype")=="bfloat16", fp16=False, gradient_checkpointing=False,
                          chat_template_kwargs={"enable_thinking": cfg.get("enable_thinking", False)}, logging_steps=1,
                          save_strategy="no", report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
                          mask_truncated_completions=True)
        class BoundedCheckpointCallback(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                save_intermediate(state.global_step)
                return control
        trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset, reward_funcs=reward, processing_class=tokenizer,
                              callbacks=[BoundedCheckpointCallback()] if planned else [])
        trainer.train()
        losses = [v["loss"] for v in trainer.state.log_history if "loss" in v]
        grad_norms = [v.get("grad_norm", 0.) for v in trainer.state.log_history if "loss" in v]
        if not any(g["nonconstant"] for g in reward_groups):
            save(root / "signal_failure.json", {"issue": "zero_variance_grpo_reward", "groups": reward_groups})
            raise ValueError("GRPO reward groups have no policy-gradient signal; revise training prompts/reward, not the utility gate")
    else:
        raise ValueError("unknown GPU operation")
    delta = sum(float((p.detach() - before[n]).float().square().sum()) for n, p in parameters.items()) ** .5
    clipping = [v.get("completions/clipped_ratio") for v in trainer.state.log_history if "loss" in v] if request["operation"] == "grpo" else []
    require_update(request["operation"], delta, grad_norms, root, reward_groups, clipping)
    model.eval()
    probe = encode_prompts([rows[0]])
    with torch.inference_mode():
        logits = model(**probe).logits[:, -1].float().cpu()
    checkpoint = root / "checkpoint"
    final_candidate = save_checkpoint(checkpoint, recipe['steps'])
    if planned:saved_candidates.append(final_candidate)
    del model, parameters, before
    if request["operation"]=="sft":del optimizer
    if request["operation"]=="grpo":del trainer
    gc.collect(); torch.cuda.empty_cache()
    reloaded, _ = load_model(checkpoint,execution)
    reloaded.eval()
    with torch.inference_mode():
        reload_error = float((reloaded(**probe).logits[:, -1].float().cpu() - logits).abs().max())
    reload_tolerance=.00001
    if reload_error > reload_tolerance:
        raise ValueError(f"adapter checkpoint reload verification failed: {reload_error}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_path(checkpoint),
            "checkpoint_candidates":saved_candidates,
            "prompt_sampling":prompt_sampling if request["operation"]=="grpo" else None,
            "optimizer_steps": len(losses), "nonzero_gradient_steps": sum(g > 0 for g in grad_norms),
            "weight_delta_l2": delta, "losses": losses, "gradient_norms": grad_norms,
            "exposure": exposure, "reward_groups": reward_groups,
            "checkpoint_format":"r2ai.lora_checkpoint.v1","model_adapter":adapter_info,"reload_tolerance":reload_tolerance,
            "merge_max_abs_error": None, "reload_max_abs_error": reload_error,
            "seconds": time.monotonic() - start, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": __import__("peft").__version__, "gpu": torch.cuda.get_device_name(0)}}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--request", required=True); p.add_argument("--result", required=True)
    a = p.parse_args()
    request = json.loads(Path(a.request).read_text())
    result = run(request)
    save(a.result, result)


if __name__ == "__main__":
    main()
