# Method mapping

| Stage | Code |
|---|---|
| Attack collection and verification | `collection/pool.py`, `portable/source_pool.py` |
| Equal-count pilots, scaling and shortlisting | `research/controller.py` |
| Full-scale trials, S1–S4 and final selection | `research/controller.py`, `research/backend.py` |
| Model loading and training | `research/model_adapter.py`, `training/` |
| Independent final evaluation | `research/full_evaluate.py` |

Paths are under `src/r2ai_autoresearch/`. All shortlisted branches complete before final selection. Accepted models become the next round's target; otherwise search retains the parent.

See [configuration](research_configuration.md), [data preparation](data_interface.md) and [acceptance](acceptance.md). The release critic is Claude Sonnet 4.6, replacing the paper's 4.5 service. Historical recipes remain in `recipes/recorded/`.
