#!/usr/bin/env python3
"""Load a Ministral LoRA independently and require a non-empty generation."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from r2ai_autoresearch.training.model_family import load_base_model, resolve_model_family


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Explain why protecting private user data matters in two sentences.")
    args = parser.parse_args()
    family = resolve_model_family(args.base, "ministral3")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model = load_base_model(
        args.base,
        family,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={index: "132GiB" for index in range(torch.cuda.device_count())},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
    ).strip()
    if not completion or len(completion) < 20:
        raise RuntimeError(f"invalid adapter completion: {completion!r}")
    result = {
        "phase": "complete",
        "base": args.base,
        "adapter": args.adapter,
        "fix_mistral_regex": True,
        "prompt": args.prompt,
        "completion": completion,
        "completion_tokens": int(output.shape[1] - inputs["input_ids"].shape[1]),
        "time": time.time(),
    }
    atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
