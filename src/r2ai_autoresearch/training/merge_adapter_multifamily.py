#!/usr/bin/env python3
"""Reload, verify, and merge a critic adapter for Qwen or Ministral3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

from r2ai_autoresearch.training.model_family import (
    load_base_model,
    resolve_model_family,
    verify_adapter_tensor_count,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    family = resolve_model_family(args.base, "auto")
    tokenizer_kwargs = {"trust_remote_code": True}
    if family.name == "ministral3":
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.base, **tokenizer_kwargs)
    base = load_base_model(
        args.base,
        family,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={index: "132GiB" for index in range(torch.cuda.device_count())},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    saved_path = args.adapter / "adapter_model.safetensors"
    saved = load_file(str(saved_path))
    verify_adapter_tensor_count(saved, family)
    actual = get_peft_model_state_dict(model)
    if set(saved) != set(actual):
        raise RuntimeError(f"adapter keys differ: saved={len(saved)}, loaded={len(actual)}")
    maximum_difference = max(
        float((actual[key].detach().cpu().to(value.dtype) - value).abs().max())
        for key, value in saved.items()
    )
    if maximum_difference != 0:
        raise RuntimeError(f"adapter reload mismatch: {maximum_difference}")

    merged = model.merge_and_unload(safe_merge=True)
    args.merged.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.merged)
    index_path = args.merged / "model.safetensors.index.json"
    result = {
        "state": "complete",
        "model_family": family.name,
        "adapter_tensors": len(saved),
        "reload_max_abs_diff": maximum_difference,
        "adapter_sha256": sha256(saved_path),
        "merged_index_sha256": sha256(index_path),
        "time": time.time(),
    }
    atomic_json(args.status, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
