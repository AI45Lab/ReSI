from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BENCHMARKS, DEFAULT_EVALUATION_ROOT, ASSET_ROOT


def _existing(root: Path, candidates: list[str]) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(
        "None of the expected data files exists: "
        + ", ".join(str(root / item) for item in candidates)
    )


def _prompt(root: Path, name: str) -> str:
    return (ASSET_ROOT / "prompts" / name).read_text(encoding="utf-8")


def _limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return rows[:limit]


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def load_harmbench(root: Path) -> list[dict[str, Any]]:
    path = _existing(
        root,
        ["safety/harmbench/harmbench.parquet", "safety/harmbench/data/harmbench.parquet"],
    )
    df = pd.read_parquet(path)
    rows = []
    for index, item in df.iterrows():
        subset = str(item["subset"])
        prompt = str(item["prompt"])
        context = _clean_scalar(item.get("context"))
        if subset == "contextual" and context:
            prompt = f"{context}\n\n{prompt}"
        rows.append(
            {
                "id": f"harmbench:{index}",
                "benchmark": "harmbench",
                "prompt": prompt,
                "reference": None,
                "metadata": {
                    "row_index": int(index),
                    "subset": subset,
                    "category": _clean_scalar(item.get("category")),
                },
            }
        )
    return rows


def load_wildjailbreak(root: Path) -> list[dict[str, Any]]:
    path = _existing(
        root,
        ["safety/wildjailbreak_test/eval.tsv", "safety/wildjailbreak_test/eval/eval.tsv"],
    )
    df = pd.read_csv(path, sep="\t")
    df = df[df["data_type"] == "adversarial_harmful"]
    rows = []
    for index, item in df.iterrows():
        rows.append(
            {
                "id": f"wildjailbreak:{index}",
                "benchmark": "wildjailbreak",
                "prompt": str(item["adversarial"]),
                "reference": None,
                "metadata": {"row_index": int(index), "data_type": item["data_type"]},
            }
        )
    return rows


def load_wj_adv_benign(root: Path) -> list[dict[str, Any]]:
    """Load WildJailbreak adversarial-benign prompts for over-refusal testing."""
    path = _existing(
        root,
        ["safety/wildjailbreak_test/eval.tsv", "safety/wildjailbreak_test/eval/eval.tsv"],
    )
    df = pd.read_csv(path, sep="\t")
    df = df[df["data_type"] == "adversarial_benign"]
    rows = []
    for index, item in df.iterrows():
        rows.append(
            {
                "id": f"wj_adv_benign:{index}",
                "benchmark": "wj_adv_benign",
                "prompt": str(item["adversarial"]),
                "reference": "safe",
                "metadata": {
                    "row_index": int(index),
                    "data_type": str(item["data_type"]),
                    "source_label": _clean_scalar(item.get("label")),
                },
            }
        )
    return rows


def load_h_cot(root: Path) -> list[dict[str, Any]]:
    path = _existing(
        root,
        [
            "safety/h_cot/train-00000-of-00001.parquet",
            "safety/h_cot/data/train-00000-of-00001.parquet",
        ],
    )
    df = pd.read_parquet(path)
    rows = []
    for index, item in df.iterrows():
        rows.append(
            {
                "id": f"h_cot:{index}",
                "benchmark": "h_cot",
                "prompt": str(item["Full_Input (H-CoT + Request)"]),
                "reference": None,
                "metadata": {
                    "row_index": int(index),
                    "category": str(item["Category"]),
                    "goal": str(item["Goal"]),
                },
            }
        )
    return rows


def load_xstest(root: Path) -> list[dict[str, Any]]:
    path = _existing(root, ["overrefusal/xstest/xstest_prompts.csv"])
    df = pd.read_csv(path)
    df = df[df["label"] == "safe"]
    rows = []
    for index, item in df.iterrows():
        rows.append(
            {
                "id": f"xstest:{int(item['id'])}",
                "benchmark": "xstest",
                "prompt": str(item["prompt"]),
                "reference": "safe",
                "metadata": {
                    "row_index": int(index),
                    "type": str(item["type"]),
                    "focus": str(item["focus"]),
                },
            }
        )
    return rows


def _format_mmlu_example(question: str, options: list[str], cot_content: str = "") -> str:
    choices = string.ascii_uppercase
    cleaned = [str(option) for option in options if str(option) != "N/A"]
    formatted = "".join(f"{choices[i]}. {option}\n" for i, option in enumerate(cleaned))
    cot = str(cot_content or "Let's think step by step.")
    if cot.startswith("A: "):
        cot = cot[3:]
    return f"Question: {question}\nOptions: {formatted}Answer: {cot}\n\n"


def load_mmlu_pro(root: Path) -> list[dict[str, Any]]:
    test_path = _existing(
        root,
        [
            "capability/mmlu_pro_1000_seed42/data/test-00000-of-00001.parquet",
            "capability/mmlu_pro/data/test-00000-of-00001.parquet",
        ],
    )
    val_path = _existing(
        root, ["capability/mmlu_pro/data/validation-00000-of-00001.parquet"]
    )
    template = _prompt(root, "mmlu_pro_official_5shot_cot.txt")
    test_df = pd.read_parquet(test_path)
    val_df = pd.read_parquet(val_path)
    validation: dict[str, list[dict[str, Any]]] = {}
    for _, item in val_df.iterrows():
        validation.setdefault(str(item["category"]), []).append(item.to_dict())

    rows = []
    for _, item in test_df.iterrows():
        category = str(item["category"])
        shots = validation.get(category, [])[:5]
        if len(shots) != 5:
            raise ValueError(f"MMLU-Pro category {category!r} has {len(shots)} validation shots")
        five_shot = "".join(
            _format_mmlu_example(
                str(shot["question"]), list(shot["options"]), str(shot["cot_content"])
            )
            for shot in shots
        )
        options = [str(option) for option in item["options"] if str(option) != "N/A"]
        formatted_options = "".join(
            f"{string.ascii_uppercase[i]}. {option}\n" for i, option in enumerate(options)
        ).rstrip()
        prompt = template.format(
            category=category,
            five_shot_examples=five_shot,
            question=str(item["question"]),
            formatted_options=formatted_options,
        )
        rows.append(
            {
                "id": f"mmlu_pro:{int(item['question_id'])}",
                "benchmark": "mmlu_pro",
                "prompt": prompt,
                "reference": str(item["answer"]),
                "metadata": {
                    "question_id": int(item["question_id"]),
                    "category": category,
                    "answer_index": int(item["answer_index"]),
                },
            }
        )
    return rows


def load_gpqa_diamond(root: Path) -> list[dict[str, Any]]:
    path = _existing(
        root, ["capability/gpqa_diamond/data/test-00000-of-00001.parquet"]
    )
    df = pd.read_parquet(path)
    rows = []
    for index, item in df.iterrows():
        rows.append(
            {
                "id": f"gpqa_diamond:{index}",
                "benchmark": "gpqa_diamond",
                "prompt": str(item["problem"]),
                "reference": str(item["solution"]),
                "metadata": {"row_index": int(index), "domain": str(item["domain"])},
            }
        )
    return rows


def load_ifeval(root: Path) -> list[dict[str, Any]]:
    path = _existing(root, ["capability/ifeval/ifeval_input_data.jsonl"])
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "id": f"ifeval:{int(item['key'])}",
                    "benchmark": "ifeval",
                    "prompt": str(item["prompt"]),
                    "reference": None,
                    "metadata": {
                        "key": int(item["key"]),
                        "instruction_id_list": item["instruction_id_list"],
                        "kwargs": item["kwargs"],
                    },
                }
            )
    return rows


LOADERS = {
    "harmbench": load_harmbench,
    "wildjailbreak": load_wildjailbreak,
    "wj_adv_benign": load_wj_adv_benign,
    "h_cot": load_h_cot,
    "xstest": load_xstest,
    "mmlu_pro": load_mmlu_pro,
    "gpqa_diamond": load_gpqa_diamond,
    "ifeval": load_ifeval,
}


EXPECTED_COUNTS = {
    "harmbench": 400,
    "wildjailbreak": 2000,
    "wj_adv_benign": 210,
    "h_cot": 50,
    "xstest": 250,
    "mmlu_pro": 1000,
    "gpqa_diamond": 198,
    "ifeval": 541,
}


def load_benchmark(
    benchmark: str,
    *,
    evaluation_root: str | Path = DEFAULT_EVALUATION_ROOT,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if benchmark not in LOADERS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choose from {', '.join(BENCHMARKS)}")
    rows = LOADERS[benchmark](Path(evaluation_root))
    return _limit(rows, limit)


def inventory(evaluation_root: str | Path = DEFAULT_EVALUATION_ROOT) -> dict[str, Any]:
    result: dict[str, Any] = {"evaluation_root": str(evaluation_root), "benchmarks": {}}
    for name in BENCHMARKS:
        try:
            rows = load_benchmark(name, evaluation_root=evaluation_root)
            count = len(rows)
            result["benchmarks"][name] = {
                "count": count,
                "expected_count": EXPECTED_COUNTS[name],
                "ok": count == EXPECTED_COUNTS[name],
            }
        except Exception as exc:
            result["benchmarks"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    result["ok"] = all(item["ok"] for item in result["benchmarks"].values())
    return result
