# Full evaluation

Run after search; final benchmark scores do not affect model selection.

```bash
bash scripts/evaluate_full.sh --config config.json --run runs/research --check
bash scripts/evaluate_full.sh --config config.json --run runs/research
```

`--check` validates inputs without starting a service. Normal execution exports the retained checkpoint, starts its local service, evaluates, then stops the service. The full installer supplies the runtimes.

## Data

Paths below are relative to `data.preparation.benchmarks`, except the separately configured X-Teaming CSV.

| Benchmark | File | Count |
|---|---|---:|
| WildJailbreak | `safety/wildjailbreak_test/eval.tsv` | 2,000 harmful |
| HarmBench | `safety/harmbench/harmbench.parquet` | 400 |
| H-CoT | `safety/h_cot/train-00000-of-00001.parquet` | 50 |
| XS-Test | `overrefusal/xstest/xstest_prompts.csv` | 250 benign |
| GPQA-Diamond | `capability/gpqa_diamond/data/test-00000-of-00001.parquet` | 198 |
| MMLU-Pro | `capability/mmlu_pro_1000_seed42/data/test-00000-of-00001.parquet` | 1,000 |
| IFEval | `capability/ifeval/ifeval_input_data.jsonl` | 541 |
| X-Teaming | `final_evaluation.xteaming.behavior_path` | 159 |

`prepare_data.sh --download` supplies WJB, H-CoT, XS-Test and IFEval. Provide:

- **HarmBench:** `prompt`, `subset`, `context`, `category` columns.
- **GPQA:** `problem` (formatted question), `solution`, `domain`.
- **MMLU-Pro:** official columns; also provide `capability/mmlu_pro/data/validation-00000-of-00001.parquet` for five-shot examples.
- **X-Teaming:** CSV with `Behavior`, `BehaviorID`, `FunctionalCategory`.

## Settings

| Field | Default / purpose |
|---|---|
| `final_evaluation.target` | Local port `8000`, served name `resi-retained` |
| `final_evaluation.output` / `.workers` | `runs/final-evaluation` / 4 |
| `evaluation.generation` | Model defaults; explicit values override them |
| `evaluation.max_new_tokens` / `.retry_max_new_tokens` | 32,768 / 65,536; one truncation retry |
| `final_evaluation.requests` | Target timeout 3,600 s; bounded retries |
| `final_evaluation.xteaming.planner` | DeepSeek-R1 endpoint/model/key variable |
| `final_evaluation.xteaming.attacker` | Qwen2.5-32B-Instruct endpoint/model/key variable |
| `final_evaluation.xteaming` | 10 plans, 7 turns, 4 TextGrad iterations |

X-Teaming attack sampling uses T=1, top-p=0.95; verification uses `services.judge`. Null planner/attacker connection fields inherit the agent endpoint. Persistent truncation and failed requests remain incomplete.

## Options and results

| Option | Purpose |
|---|---|
| `--checkpoint PATH` | Evaluate a complete HF model or framework checkpoint instead of `--run` |
| `--external-target` | Use your configured service for that checkpoint |
| `--export-dir PATH` | Relocate exported weights |
| `--limit N` | Reduced check, not full benchmark results |

GPT-4o at T=0 applies the [safety prompt](prompts/safety_judge.txt); score 5 is unsafe. XS-Test uses its [refusal classifier](prompts/benign_compliance_judge.txt); GPQA/MMLU report accuracy and IFEval runs official checks.

Read `summary.json`, `metrics.csv` and per-example predictions/judgments. Repeat to resume unchanged inputs; changed settings, data or checkpoint require a new output directory. Keep development/test overlap explicit when reporting results.
