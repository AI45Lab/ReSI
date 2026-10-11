# Third-party components

ReSI project code uses the repository MIT license. Dependency code, prompts, and datasets retain their original terms.

- The bundled IFEval checker comes from Google Research; its Apache 2.0 license is in `IFEVAL_LICENSE.txt`.
- `setup_verl.sh` installs VERL commit `c4b389adadc58ce51cb2b63e70df497ca166d77f` from `https://github.com/verl-project/verl` and applies the bundled historical runtime patch. Its Apache 2.0 license is in `VERL_LICENSE.txt`. The patch preserves upstream notices.
- X-Teaming planner/attacker YAML prompts are not redistributed in the source archive. `scripts/setup_xteaming.py` retrieves upstream revision `db964e6630b6a69b78e093705c89fd34865b6c27` with SHA-256 verification; upstream terms apply. The scoring rubric is attributed to X-Teaming and is not a new ReSI scoring method.
- JailbreakSkill is an optional pinned upstream installation, not MIT-relicensed project code. Its install script checks revision `3d164bfc85ea9a39d02e41891734528a8520ad9d`.
- WildJailbreak, H-CoT, XS-Test, HarmBench, GPQA-Diamond, MMLU-Pro and DOLCI are independently distributed datasets. The repository bundles authored demonstration records, not their full data. See `DOLCI.md` and the pinned download metadata.

`docs/migration_sources.json` records source fingerprints and integration changes. It is provenance only; runtime imports never read those historical locations.
