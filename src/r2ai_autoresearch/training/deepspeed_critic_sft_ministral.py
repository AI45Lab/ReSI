#!/usr/bin/env python3
"""Manifest-bound Ministral-3 critic LoRA SFT with adapter-only saves."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.integrations import HfDeepSpeedConfig

from r2ai_autoresearch.training.deepspeed_critic_sft_a3 import (
    DynamicCollator,
    atomic_json,
    sha256,
)
from r2ai_autoresearch.training.model_family import (
    load_base_model,
    resolve_model_family,
    verify_adapter_tensor_count,
)


class RankLossAuditTrainer(Trainer):
    """Optionally expose the pre-reduction loss from each distributed rank."""

    def __init__(self, *args, audit_rank_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_rank_loss = audit_rank_loss
        self._rank_loss_printed = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        loss = result[0] if return_outputs else result
        if self.audit_rank_loss and not self._rank_loss_printed:
            print(
                "RANK_LOCAL_LOSS "
                f"rank={os.environ.get('RANK', '0')} loss={loss.detach().float().item():.9f}",
                flush=True,
            )
            self._rank_loss_printed = True
        return result


def main(dataset_class=None, encoding_metadata=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--parent-adapter",
        help="Continue an existing compatible LoRA instead of initializing a new one.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--audit-rank-loss", action="store_true")
    parser.add_argument("--lazy-tokenize", action="store_true", help="Compatibility flag; official lazy encoding is always used.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from r2ai_autoresearch.training.ministral_sft_encoding import require_reasoning_channel
    require_reasoning_channel()
    # CLI paths are authoritative; stale shell environment must not select another tokenizer.
    os.environ['R2AI_MINISTRAL_BASE'] = args.base
    os.environ['R2AI_MINISTRAL_OUTPUT'] = args.output
    os.environ['R2AI_MINISTRAL_ASSISTANT_CHANNEL'] = 'reasoning'
    if dataset_class is None:
        from r2ai_autoresearch.training.deepspeed_critic_sft_ministral_official import OfficialEncodedCriticDataset
        dataset_class = OfficialEncodedCriticDataset

    manifest = json.loads(Path(args.data_manifest).read_text())
    assert manifest["state"] == "frozen_train_ready"
    assert manifest["formal_test_data_used"] is False
    assert manifest["system_role_present"] is False
    assert manifest["loss_mask"] == "assistant_target_only"
    assert manifest["max_sequence_length"] == args.max_length
    assert manifest["model_family"] == "ministral3"
    assert manifest["reasoning_protocol"] == "mistral_square_think_v1"
    assert manifest["artifacts"]["train.parquet"] == sha256(args.data)

    deepspeed_lifetime_guard = None
    if args.deepspeed:
        ds_config = json.loads(Path(args.deepspeed).read_text())
        ds_config["train_micro_batch_size_per_gpu"] = args.microbatch
        ds_config["gradient_accumulation_steps"] = args.grad_accum
        ds_config["train_batch_size"] = args.microbatch * args.grad_accum * int(os.environ.get("WORLD_SIZE", "1"))
        deepspeed_lifetime_guard = HfDeepSpeedConfig(ds_config)
    family = resolve_model_family(args.base, "ministral3")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset_class(args.data, tokenizer, args.max_length, args.limit)
    started = time.time()
    model = load_base_model(
        args.base,
        family,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    parent_state = None
    if args.parent_adapter:
        parent = Path(args.parent_adapter)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (parent / filename).is_file():
                raise RuntimeError(f"parent adapter missing {filename}: {parent}")
        parent_state = load_file(str(parent / "adapter_model.safetensors"))
        verify_adapter_tensor_count(parent_state, family)
        # GRPO exports serialize a flat target-module list.  On the multimodal
        # Ministral wrapper, loading that list verbatim also attaches fresh,
        # missing LoRAs to the vision tower.  Reapply the model-family contract
        # so continuation contains exactly the 560 text-model tensors that were
        # present in the parent.
        parent_config = PeftConfig.from_pretrained(parent)
        parent_config.target_modules = family.lora_target
        model = PeftModel.from_pretrained(
            model, parent, config=parent_config, is_trainable=True
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=family.lora_target,
            ),
        )
    model.enable_input_require_grads()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not all("lora_" in name for name in trainable):
        raise RuntimeError("non-LoRA trainable parameters detected")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        do_train=True,
        per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # Transformers 5.x accepts a fractional warmup through warmup_steps
        # and no longer exposes the warmup_ratio constructor argument.
        warmup_steps=args.warmup_ratio,
        optim="adamw_torch_fused",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_cache=False,
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        save_strategy="no" if not args.save_steps else "steps",
        save_steps=args.save_steps or 500,
        save_only_model=True,
        train_sampling_strategy="random" if args.lazy_tokenize else "group_by_length",
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=False,
    )
    trainer = RankLossAuditTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DynamicCollator(tokenizer.pad_token_id),
        audit_rank_loss=args.audit_rank_loss,
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        weights = output / "adapter" / "adapter_model.safetensors"
        state = load_file(str(weights))
        verify_adapter_tensor_count(state, family)
        parent_delta_norm = None
        if parent_state is not None:
            if set(state) != set(parent_state):
                raise RuntimeError("saved adapter tensor keys differ from parent")
            squared_delta = sum(
                float(torch.sum((state[name].float() - parent_state[name].float()) ** 2))
                for name in state
            )
            parent_delta_norm = squared_delta ** 0.5
            if parent_delta_norm <= 0:
                raise RuntimeError("continued adapter is identical to parent")
        b_nonzero = sum(
            torch.count_nonzero(tensor).item()
            for name, tensor in state.items()
            if "lora_B" in name
        )
        logs = [entry for entry in trainer.state.log_history if "loss" in entry]
        if b_nonzero <= 0 or not any(float(entry.get("grad_norm", 0)) > 0 for entry in logs):
            raise RuntimeError("no nonzero optimizer update evidence")
        summary = {
            "state": "adapter_saved_pending_reload",
            "base": args.base,
            "parent_adapter": args.parent_adapter,
            "parent_adapter_sha256": (
                sha256(Path(args.parent_adapter) / "adapter_model.safetensors")
                if args.parent_adapter
                else None
            ),
            "parent_to_candidate_delta_norm": parent_delta_norm,
            "model_family": family.name,
            "fix_mistral_regex": True,
            "lora_target": family.lora_target,
            "expected_adapter_tensors": family.expected_adapter_tensors,
            "adapter_tensors": len(state),
            "adapter_sha256": sha256(weights),
            "data": args.data,
            "data_sha256": sha256(args.data),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "microbatch": args.microbatch,
            "grad_accum": args.grad_accum,
            "global_batch": int(os.environ.get("WORLD_SIZE", "1")) * args.microbatch * args.grad_accum,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
            "distributed_backend": "deepspeed" if args.deepspeed else "ddp",
            "lazy_tokenize": True,
            "encoding_backend": "mistral_common_finetuning",
            "encoding_code_sha256": sha256(Path(__file__).with_name("ministral_sft_encoding.py")),
            "assistant_channel": "reasoning",
            "system_prompt_source": str(Path(args.base) / "SYSTEM_PROMPT.txt"),
            "full_sequence_matches_production_tokenizer_contract": True,
            "dataset_encoder": dataset_class.__name__,
            "steps": trainer.state.global_step,
            "train_rows": len(dataset),
            "elapsed_seconds": time.time() - started,
            "trainer_metrics": result.metrics,
            "log_history": logs,
        }
        summary.update(encoding_metadata or {})
        atomic_json(output / "training_run_manifest.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
