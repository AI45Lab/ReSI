#!/usr/bin/env python3
"""Strict adapter reload verification for supported R2AI model families."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import load_file

from r2ai_autoresearch.training.model_family import (
    load_base_model,
    resolve_model_family,
    verify_adapter_tensor_count,
)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--model-family", choices=("qwen36", "ministral3"), required=True)
    args = parser.parse_args()
    weights = Path(args.adapter) / "adapter_model.safetensors"
    saved = load_file(weights)
    family = resolve_model_family(args.base, args.model_family)
    verify_adapter_tensor_count(saved, family)
    model = load_base_model(
        args.base,
        family,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={index: int(torch.cuda.get_device_properties(index).total_memory * .8) for index in range(torch.cuda.device_count())},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    reloaded = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    actual = get_peft_model_state_dict(reloaded)
    if set(saved) != set(actual):
        raise RuntimeError(
            f"adapter key mismatch: missing={sorted(set(saved) - set(actual))[:5]}, "
            f"extra={sorted(set(actual) - set(saved))[:5]}"
        )
    max_diff = max(
        float((actual[name].detach().cpu().to(expected.dtype) - expected).abs().max())
        for name, expected in saved.items()
    )
    if max_diff != 0.0:
        raise RuntimeError(f"strict reload mismatch: max_abs_diff={max_diff}")
    result = {
        "phase": "complete",
        "adapter": args.adapter,
        "base": args.base,
        "model_family": family.name,
        "expected_adapter_tensors": family.expected_adapter_tensors,
        "adapter_tensors": len(saved),
        "reload_tensors": len(actual),
        "reload_max_abs_diff": max_diff,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "time": time.time(),
    }
    atomic(Path(args.adapter) / "reload_evidence.json", result)
    atomic(args.status, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
