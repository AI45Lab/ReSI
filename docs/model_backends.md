# Model backends

Set model and training fields in `config.json`; all backends share the search controller and gate.

| Field | Setting |
|---|---|
| `model.path` | Model directory, including tokenizer assets |
| `runtime.gpu_ids` | Visible GPUs, e.g. `0,1` |
| `model.dtype` | `float32` or `bfloat16` |
| `model.backend` | `single_gpu` or `hf_sharded` (layer sharding) |
| `training.lora.target_modules` | Explicit projections, or `model_default` for native text targets |
| `model.trust_remote_code` | Opt-in for custom architecture code |

## Training engines

| Execution | `training.sft.engine` | `training.grpo.engine` |
|---|---|---|
| Single-process | `portable` | `portable` |
| Native Qwen / Ministral | `historical_deepspeed` | `historical_verl` |

The [full installer](../README.md#installation) creates separate SFT and VERL/vLLM environments. `hf_sharded` alone does not enable distributed optimizer or expert-parallel training.

`training.batch_size` counts SFT examples or GRPO completions. Native SFT requires divisibility by GPU count × microbatch; GRPO divides by `num_generations` for prompt groups. Tensor-parallel size must divide GPU count.

## Checkpoint export

Trials save `adapter/`, model assets and `r2ai_checkpoint.json`. Keep the referenced parent checkpoints and their ancestry.

```bash
bash scripts/export_model.sh --config config.json --run runs/research --output runs/exported
```

The [full evaluator](formal_evaluation.md) exports automatically. Historical model settings are in `recipes/recorded/`; [acceptance](acceptance.md) lists tested paths and limits.
