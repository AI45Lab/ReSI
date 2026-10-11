# Data preparation

Set the following fields under `data.preparation`. Paths are relative to `config.json`.

| Field | Input / output |
|---|---|
| `safety_pool` | Original safety requests, JSON |
| `training` | Attack and replay rows, JSON |
| `training_seeds` | Remaining seeds after validation split |
| `benchmarks` | Local [benchmark files](formal_evaluation.md#data) |
| `validation_counts` | WJB holdout 128; H-CoT, benign XS-Test and IFEval 50 each |

## Prepare inputs

Original safety-pool format; use unique IDs and enough harmful examples for training and validation:

```json
{"rows": [{"id":"seed-0001", "prompt":"Original request", "kind":"harmful"}]}
```

Reserve validation before generating attacks:

```bash
bash scripts/prepare_data.sh --config config.json --split-only
```

Generate training attacks only from `training_seeds`. Imported training rows use this format:

```json
{"rows": [
  {"id":"attack-0001", "seed_id":"seed-0001", "source":"darwin", "split":"train", "kind":"harmful", "prompt":"Attack text"}
]}
```

Use `kind: "benign"` for benign safety rows, `mixture_role: "replay"` for replay, and source IDs matching `data.sources`. Every safety row requires its original `seed_id`.

Build `data.path`:

```bash
bash scripts/prepare_data.sh --config config.json --download
```

Omit `--download` for local benchmark resources. Preparation rejects training/validation seed overlap and exact benchmark-prompt matches. `data.original_safety_source` includes verified original-prompt successes in the first-round shared base. All attack pools are rechecked against each round's model; `data.max_candidates_per_source` must cover the complete pool.

## General-task data

Set `data.preparation.general`:

| Field | Setting |
|---|---|
| `source` | `dolci` (default), `custom`, or `manifest` for existing utility rows |
| `dataset`, `revision` | Pinned DOLCI source in `config.json` |
| `max_examples` | 20,000 retained examples by default |
| `path` | Custom JSON/JSONL file |
| `additional_sources` | Extra datasets, e.g. `[{"name":"math","path":"data/math.jsonl","max_examples":5000}]` |

Custom rows require `prompt` and `target`; JSON accepts an array or `{"rows": [...]}`. JSONL uses one row per line:

```json
{"prompt":"Question", "target":"Reference answer"}
```

DOLCI retains single-turn, non-safety examples and published answers. Duplicate/evaluation prompts are excluded. S3 uses general data for SFT only; loading the pool does not alter the initial mixture.

## Attack generation

Copy selected entries from [collection.example.json](../configs/collection.example.json) into `data.providers`. Configure `seed_source`, service settings and `attempts_per_seed`.

| Provider | Setup |
|---|---|
| `darwin_p2` | Generator endpoint for bounded strategy-catalog search |
| `jailbreak_skill` | `bash scripts/setup_attack_sources.sh`; select skill names and rewrite service |
| `magic` | Endpoint serving a trained MAGIC attacker |

Configured providers generate against the current parent; other sources use imported attacks. The collection cap must cover all seeds and attempts. Source labels alone do not invoke generators.
