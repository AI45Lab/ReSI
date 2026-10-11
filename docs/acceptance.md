# Acceptance

Engineering checks completed on 2026-10-11; detailed measurements are in [acceptance_receipts.json](acceptance_receipts.json).

| Check | Result |
|---|---|
| Tests | 375 passed, 1 skipped; all 150 public-package tests passed |
| A3 | Qwen observations, paired-data construction, weighting, SFT and evaluation passed |
| Search/resume | Six pilots and one probe attempted; parent retained; resume reused completed receipts |
| Training components | General-data mixing, saved-checkpoint selection and SFT-to-GRPO continuation passed |
| Native Qwen / Ministral | Short distributed SFT/GRPO, parameter updates and strict reload passed |
| Evaluation | Seven static runners at one example each; X-Teaming at one behavior and two turns |
| Installation/package | Fresh GPU installation, SFT, wheel build, syntax, links, hashes and secret scans passed |

These checks do not reproduce paper scores. Full-scale selection and S1–S4 combine controlled tests with separate real component runs; the latest small-model search retained its parent. New DeepSeek/Safin training in this runtime and live MAGIC generation were outside the tested scope. The DARWIN provider implements bounded strategy-catalog search.
