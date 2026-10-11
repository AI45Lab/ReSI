# Configuration

Edit [config.json](../config.json). Paths are relative to this file. Supply API keys through the environment variables named by `api_key_env`.

## Model and services

| Field | Set to |
|---|---|
| `model.path` | Your local model directory; alternatively set `model.id` and pinned `model.revision` for download |
| `runtime.gpu_ids` | GPU IDs, e.g. `0,1` |
| `data.path` | Prepared [data manifest](data_interface.md) |
| `runtime.output` | Search output; default `runs/research` |
| `services.agent` / `.critic` / `.judge` | `base_url`, `model`, `api_key_env`, `timeout_seconds` |
| `training.sft.engine` / `training.grpo.engine` | [Training backends](model_backends.md) |

Agent/judge default to GPT-4o; critic to `claude-sonnet-4-6`, all at T=0. Replace the template model with your target model.

## Search and training defaults

| Field | Default |
|---|---|
| `search.methods` | `critic_sft`, `a3`, `direct_grpo` |
| `search.n0` | 512 attacks per source |
| `search.scaling_probe_samples` / `.max_scaling_probes` | 2,048 / 1 per round |
| `search.shortlist_size` | 2 |
| `search.max_refinements_per_phase` | 2 revisions per phase; reset for S4 |
| `search.max_rounds` | 3; accepts 1–3 |
| `training.sft.pilot_steps` / `.full_steps` | 128 / 512 |
| `training.grpo.pilot_steps` / `.full_steps` | 8 / 62 |
| `training.sft.learning_rate` / `training.grpo.learning_rate` | 2e-5 / 1e-6 |
| `training.batch_size` / `.max_sequence_length` | 8 / 4,096 |
| `training.lora` | Rank 16, alpha 32; configure architecture-compatible targets |

## Gate and skills

Safety compares with the current parent; BC/IF compare with the initial model. Values below are absolute proportions: **0.03 = 3 percentage points**.

| Field | Default / action |
|---|---|
| `evaluation.supplemental_threshold` | 0.05: below this WJB ASR, H-CoT determines safety gain |
| `evaluation.wjb_asr_tolerance` | 0.02: allowed WJB increase when H-CoT is active |
| `evaluation.benign_compliance_tolerance` / `.utility_tolerance` | 0.03 / 0.03 |
| `skills.S1.max_sources` | 2; compose promising sources |
| `skills.S2.benign_repeats` | `[2,3]`; sample verified benign counterparts |
| `skills.S3.general_weights` | `[0.15,0.20]`; SFT general-data shares |
| `skills.S3.lr_multipliers` / `.update_multipliers` | `[0.5]` / `[0.5]`; separate revisions |
| `skills.S4.enabled` | `true`; GRPO continuation from an SFT checkpoint |
| `skills.S4.allow_ineligible_sft` | `true`; recovery requires safety gain ≥0.05, BC/IF drops ≤0.06 and the original safety guard |

S4 recovery thresholds are configurable as `min_safety_gain`, `max_benign_compliance_drop` and `max_utility_drop`. Final selection always uses the original gate and retains eligible SFT candidates.

## Method-specific settings

| Section | Controls |
|---|---|
| `a3.safin` | Behaviors 16; hypotheses 10 × 6; pairs/hypothesis 20; train/validation/OOD 70/20/10; `target_max_tokens=8192`; weighted sampling via `weight_bands`, `logical_epochs=4` |
| `training.grpo` | Group size 8, T=1, top-p=0.95, top-k=0; screening keeps prompts with mixed 0/1 rewards |
| `training.checkpoint_selection` | Optional intermediate selection; disabled, up to 3 checkpoints when enabled; portable workers only |
| `evaluation.diagnostics` | Training-review alert for ≥3 abnormal responses and ≥20% abnormal outputs |
| `final_evaluation` | [Full benchmark configuration](formal_evaluation.md) |

## Run and resume

```bash
bash scripts/run.sh --config config.json --check
bash scripts/run.sh --config config.json
bash scripts/report.sh --run runs/research
```

`--check` makes no API/GPU calls. Read `summary.json` and the generated `report/trials.csv`. Repeat the command to resume; changed configuration, data, model or execution code requires a new output directory.
