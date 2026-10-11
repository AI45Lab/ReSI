#!/usr/bin/env python3
"""Manifest-bound distributed Qwen Critic LoRA SFT via Trainer + DeepSpeed."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.integrations import HfDeepSpeedConfig
from .historical_worker import encode_assistant_target


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class EncodedCriticDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length, limit=0):
        frame = pd.read_parquet(parquet_path)
        if limit:
            frame = frame.iloc[:limit]
        self.rows = []
        self.total_tokens = 0
        self.total_label_tokens = 0
        for row in frame.itertuples(index=False):
            messages = list(row.messages)
            encoded = encode_assistant_target(tokenizer, messages, max_length)
            self.rows.append(encoded)
            self.total_tokens += encoded['length']
            self.total_label_tokens += sum(x != -100 for x in encoded['labels'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class DynamicCollator:
    def __init__(self, pad_token_id, pad_multiple=8):
        self.pad_token_id = pad_token_id
        self.pad_multiple = pad_multiple

    def __call__(self, features):
        longest = max(len(item["input_ids"]) for item in features)
        longest = math.ceil(longest / self.pad_multiple) * self.pad_multiple
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = longest - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad)
            result["attention_mask"].append(item["attention_mask"] + [0] * pad)
            result["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-reentrant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    from r2ai_autoresearch.training.ministral_sft_encoding import reject_legacy_ministral_base
    reject_legacy_ministral_base(args.base)

    manifest = json.load(open(args.data_manifest))
    assert manifest["state"] == "frozen_train_ready"
    assert manifest["max_sequence_length"] == args.max_length
    assert manifest["system_role_present"] is False
    assert manifest["loss_mask"] == "assistant_target_only"
    assert manifest["formal_test_data_used"] is False
    assert manifest["artifacts"]["train.jsonl"] == sha256(Path(args.data_manifest).parent / "train.jsonl")
    assert manifest["artifacts"]["train.parquet"] == sha256(args.data)

    # This object must stay alive and must be created before from_pretrained so
    # ZeRO-3 partitions the frozen base during loading instead of materializing
    # one full CPU copy per rank.
    early_ds_config = json.load(open(args.deepspeed))
    early_ds_config["train_micro_batch_size_per_gpu"] = args.microbatch
    early_ds_config["gradient_accumulation_steps"] = args.grad_accum
    # zero.Init runs before torch.distributed initialization and therefore sees
    # world_size=1. Trainer later reconciles the real 8-rank global batch.
    early_ds_config["train_batch_size"] = args.microbatch * args.grad_accum
    hf_deepspeed_config = HfDeepSpeedConfig(early_ds_config)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.time()
    dataset = EncodedCriticDataset(args.data, tokenizer, args.max_length, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model.config.use_cache = False
    target = (r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
              r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
              r"mlp\.(?:gate_proj|up_proj|down_proj))$")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=target
    ))
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"), do_train=True,
        per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1.0, max_steps=args.max_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch_fused", weight_decay=0.0,
        bf16=True, tf32=True, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": args.checkpoint_reentrant}, use_cache=False,
        logging_steps=1, logging_first_step=True, report_to="none",
        save_strategy="no" if not args.save_steps else "steps",
        save_steps=args.save_steps or 500, save_only_model=True,
        train_sampling_strategy="group_by_length", length_column_name="length",
        remove_unused_columns=False, dataloader_num_workers=2,
        dataloader_pin_memory=True, dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2, deepspeed=args.deepspeed,
        seed=args.seed, data_seed=args.seed, disable_tqdm=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DynamicCollator(tokenizer.pad_token_id)
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        weights = output / "adapter" / "adapter_model.safetensors"
        if not weights.exists():
            raise RuntimeError("adapter-only save missing adapter_model.safetensors")
        state = load_file(str(weights))
        b_nonzero = sum(torch.count_nonzero(tensor).item() for name, tensor in state.items() if "lora_B" in name)
        if b_nonzero <= 0:
            raise RuntimeError("no nonzero LoRA-B update")
        logs = [entry for entry in trainer.state.log_history if "loss" in entry]
        if not logs or not any(float(entry.get("grad_norm", 0)) > 0 for entry in logs):
            raise RuntimeError("no nonzero grad_norm in Trainer log")
        elapsed = time.time() - started
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        samples_seen = min(len(dataset), trainer.state.global_step * args.microbatch * args.grad_accum * world_size)
        train_runtime = float(result.metrics["train_runtime"])
        summary = {
            "state": "adapter_saved_pending_reload", "base": args.base,
            "data": args.data, "data_sha256": sha256(args.data),
            "data_manifest": args.data_manifest,
            "data_manifest_sha256": sha256(args.data_manifest),
            "deepspeed_config": args.deepspeed,
            "deepspeed_config_sha256": sha256(args.deepspeed),
            "world_size": world_size, "microbatch": args.microbatch,
            "gradient_accumulation_steps": args.grad_accum,
            "global_batch": world_size * args.microbatch * args.grad_accum,
            "gradient_checkpointing": args.gradient_checkpointing,
            "checkpoint_reentrant": args.checkpoint_reentrant,
            "steps": trainer.state.global_step, "samples_seen": samples_seen,
            "elapsed_seconds_including_load_and_tokenize": elapsed,
            "trainer_metrics": result.metrics, "log_history": logs,
            "dataset_total_tokens": dataset.total_tokens,
            "dataset_total_label_tokens": dataset.total_label_tokens,
            "dataset_tokens_per_second": dataset.total_tokens / train_runtime,
            "dataset_label_tokens_per_second": dataset.total_label_tokens / train_runtime,
            "adapter_sha256": sha256(weights), "lora_b_nonzero_elements": b_nonzero,
        }
        atomic_json(output / "training_run_manifest.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
