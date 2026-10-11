from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import check_endpoint, generate_manifest
from .config import (
    BENCHMARKS,
    DEFAULT_EVALUATION_ROOT,
    JUDGE_BENCHMARKS,
    Endpoint,
    GenerationConfig,
    dump_json,
    generation_for_benchmark,
)
from .data import inventory, load_benchmark
from .io import append_jsonl, latest_records, read_jsonl, write_jsonl
from .scoring import (
    judge_prediction,
    judge_predictions,
    prepare_judge,
    score_benchmark,
    valid_judge_record,
)


def _manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _manifest_benchmark(rows: list[dict[str, Any]]) -> str:
    benchmarks = {str(row.get("benchmark")) for row in rows}
    if len(benchmarks) != 1:
        raise ValueError(
            f"Manifest must contain exactly one benchmark, got {sorted(benchmarks)}"
        )
    return next(iter(benchmarks))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _generation_extra_body(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    reserved = {"model", "messages", "temperature", "top_p", "max_tokens", "seed"}
    overlap = sorted(reserved.intersection(parsed))
    if overlap:
        raise ValueError(
            "--extra-body-json cannot override explicit generation fields: "
            + ", ".join(overlap)
        )
    return parsed


def cmd_inventory(args: argparse.Namespace) -> int:
    result = inventory(args.evaluation_root)
    _print_json(result)
    return 0 if result["ok"] else 1


def cmd_endpoint_check(args: argparse.Namespace) -> int:
    endpoint = Endpoint.from_file(args.endpoint)
    result = check_endpoint(endpoint)
    result["endpoint"] = endpoint.public_dict()
    result["configured_model_present"] = endpoint.model in result["models"]
    _print_json(result)
    return 0 if result["configured_model_present"] else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    rows = load_benchmark(
        args.benchmark,
        evaluation_root=args.evaluation_root,
        limit=args.limit,
    )
    count = write_jsonl(args.output, rows)
    _print_json({"benchmark": args.benchmark, "count": count, "output": str(args.output)})
    return 0


def _generation(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        extra_body=_generation_extra_body(args.extra_body_json),
        retry_max_tokens=getattr(args,'retry_max_tokens',None),
    )


def cmd_generate(args: argparse.Namespace) -> int:
    endpoint = Endpoint.from_file(args.endpoint)
    rows = _manifest(args.manifest)
    benchmark = _manifest_benchmark(rows)
    base_generation = _generation(args)
    generation = generation_for_benchmark(benchmark, base_generation)
    result = generate_manifest(
        rows,
        endpoint,
        args.output,
        generation,
        workers=args.workers,
        resume=args.resume,
    )
    result["endpoint"] = endpoint.public_dict()
    result["benchmark"] = benchmark
    result["base_max_tokens"] = base_generation.max_tokens
    result["resolved_max_tokens"] = generation.max_tokens
    _print_json(result)
    return 0 if result["failed"] == 0 else 2


def cmd_judge(args: argparse.Namespace) -> int:
    endpoint = Endpoint.from_file(args.endpoint)
    rows = _manifest(args.manifest)
    result = judge_predictions(
        rows,
        args.predictions,
        endpoint,
        args.output,
        evaluation_root=args.evaluation_root,
        workers=args.workers,
        resume=args.resume,
    )
    result["endpoint"] = endpoint.public_dict()
    _print_json(result)
    return 0 if result["failed"] == 0 else 2


def cmd_score(args: argparse.Namespace) -> int:
    rows = _manifest(args.manifest)
    metrics = score_benchmark(
        rows,
        args.predictions,
        args.output_dir,
        judges_path=args.judges,
    )
    _print_json(metrics)
    return 0


def _generate_with_streaming_judge(
    rows: list[dict[str, Any]],
    target: Endpoint,
    predictions_path: Path,
    generation: GenerationConfig,
    generation_workers: int,
    judge: Endpoint,
    judges_path: Path,
    evaluation_root: str | Path,
    judge_workers: int,
    resume: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    benchmark = _manifest_benchmark(rows)
    template, judge_generation = prepare_judge(benchmark, evaluation_root)
    completed_records = (
        latest_records(judges_path, require_response=True) if resume else {}
    )
    completed = {
        item_id: record
        for item_id, record in completed_records.items()
        if valid_judge_record(benchmark, record)
    }
    scheduled = set(completed)
    counts = {
        "submitted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": len(completed),
    }
    lock = threading.Lock()
    mode = "a" if resume else "w"
    judges_path.parent.mkdir(parents=True, exist_ok=True)

    with judges_path.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, judge_workers)) as pool:

            def finish_judge(future, row: dict[str, Any]) -> None:
                record = {
                    "id": row["id"],
                    "benchmark": row.get("benchmark"),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    record.update(future.result())
                    succeeded = True
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    succeeded = False
                with lock:
                    counts["succeeded" if succeeded else "failed"] += 1
                    append_jsonl(handle, record)

            def submit_judge(
                row: dict[str, Any], prediction: dict[str, Any]
            ) -> None:
                item_id = str(row["id"])
                with lock:
                    if item_id in scheduled:
                        return
                    scheduled.add(item_id)
                    counts["submitted"] += 1
                future = pool.submit(
                    judge_prediction,
                    row,
                    str(prediction["response"]),
                    judge,
                    template,
                    judge_generation,
                )
                future.add_done_callback(
                    lambda completed_future, source=row: finish_judge(
                        completed_future, source
                    )
                )

            if resume:
                existing_predictions = latest_records(
                    predictions_path, require_response=True
                )
                for row in rows:
                    prediction = existing_predictions.get(str(row["id"]))
                    if prediction is not None:
                        submit_judge(row, prediction)

            generation_counts = generate_manifest(
                rows,
                target,
                predictions_path,
                generation,
                workers=generation_workers,
                resume=resume,
                success_callback=submit_judge,
            )
    return generation_counts, counts


def cmd_suite(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    target = Endpoint.from_file(args.endpoint)
    judge = Endpoint.from_file(args.judge_endpoint) if args.judge_endpoint else None
    base_generation = _generation(args)
    generation_by_benchmark = {
        benchmark: generation_for_benchmark(benchmark, base_generation)
        for benchmark in args.benchmarks
    }
    if args.max_model_len is not None:
        required_output_limit = max(
            generation.max_tokens for generation in generation_by_benchmark.values()
        )
        if args.max_model_len < required_output_limit:
            raise ValueError(
                f"--max-model-len={args.max_model_len} is below the largest "
                f"resolved output limit {required_output_limit}"
            )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_root": str(Path(args.evaluation_root).resolve()),
        "target_endpoint": target.public_dict(),
        "judge_endpoint": judge.public_dict() if judge else None,
        "generation": base_generation.__dict__,
        "generation_by_benchmark": {
            benchmark: generation.__dict__
            for benchmark, generation in generation_by_benchmark.items()
        },
        "max_model_len": args.max_model_len,
        "length_policy": {
            "gpqa_diamond_max_tokens_multiplier": 2,
            "gpqa_diamond_max_model_len_multiplier": 2,
        },
        "limit": args.limit,
        "benchmarks": args.benchmarks,
        "stream_judge": args.stream_judge,
        "benchmark_workers": args.benchmark_workers,
    }
    dump_json(run_dir / "resolved_run.json", metadata)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "benchmarks": {},
    }
    def evaluate_benchmark(benchmark: str) -> dict[str, Any]:
        bench_dir = run_dir / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        rows = load_benchmark(
            benchmark,
            evaluation_root=args.evaluation_root,
            limit=args.limit,
        )
        manifest_path = bench_dir / "manifest.jsonl"
        predictions_path = bench_dir / "predictions.jsonl"
        judges_path = bench_dir / "judges.jsonl"
        write_jsonl(manifest_path, rows)
        generation = generation_by_benchmark[benchmark]
        judge_counts = None
        if (
            benchmark in JUDGE_BENCHMARKS
            and judge is not None
            and args.stream_judge
        ):
            generation_counts, judge_counts = _generate_with_streaming_judge(
                rows,
                target,
                predictions_path,
                generation,
                args.workers,
                judge,
                judges_path,
                args.evaluation_root,
                args.judge_workers,
                args.resume,
            )
        else:
            generation_counts = generate_manifest(
                rows,
                target,
                predictions_path,
                generation,
                workers=args.workers,
                resume=args.resume,
            )
        item_summary: dict[str, Any] = {
            "generation": generation_counts,
            "generation_config": generation.__dict__,
        }
        if benchmark in JUDGE_BENCHMARKS:
            if judge is None:
                item_summary["status"] = "generated_only_missing_judge_endpoint"
                return item_summary
            if judge_counts is None:
                judge_counts = judge_predictions(
                    rows,
                    predictions_path,
                    judge,
                    judges_path,
                    evaluation_root=args.evaluation_root,
                    workers=args.judge_workers,
                    resume=args.resume,
                )
            item_summary["judge"] = judge_counts
            metrics = score_benchmark(
                rows,
                predictions_path,
                bench_dir,
                judges_path=judges_path,
            )
        else:
            metrics = score_benchmark(rows, predictions_path, bench_dir)
        item_summary["metrics"] = metrics
        item_summary["status"] = "complete"
        return item_summary

    had_error = False
    benchmark_workers = min(max(1, args.benchmark_workers), len(args.benchmarks))
    with ThreadPoolExecutor(max_workers=benchmark_workers) as pool:
        futures = {
            pool.submit(evaluate_benchmark, benchmark): benchmark
            for benchmark in args.benchmarks
        }
        for future in as_completed(futures):
            benchmark = futures[future]
            try:
                summary["benchmarks"][benchmark] = future.result()
            except Exception as exc:
                had_error = True
                summary["benchmarks"][benchmark] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            dump_json(run_dir / "summary.json", summary)
    _print_json(summary)
    return 2 if had_error else 0


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry-max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="Additional OpenAI-compatible sampling fields as a JSON object.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m r2ai_autoresearch.formal_eval",
        description="Endpoint-driven, resumable R2AI evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inv = subparsers.add_parser("inventory", help="Validate benchmark files and counts")
    inv.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION_ROOT))
    inv.set_defaults(func=cmd_inventory)

    endpoint_check = subparsers.add_parser("endpoint-check", help="Probe an endpoint")
    endpoint_check.add_argument("--endpoint", required=True)
    endpoint_check.set_defaults(func=cmd_endpoint_check)

    prepare = subparsers.add_parser("prepare", help="Build one benchmark manifest")
    prepare.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    prepare.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION_ROOT))
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=cmd_prepare)

    generate = subparsers.add_parser("generate", help="Run target-model inference")
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--endpoint", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    _add_generation_args(generate)
    generate.set_defaults(func=cmd_generate)

    judge = subparsers.add_parser("judge", help="Judge safety or XSTest predictions")
    judge.add_argument("--manifest", required=True)
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--endpoint", required=True)
    judge.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION_ROOT))
    judge.add_argument("--output", required=True)
    judge.add_argument("--workers", type=int, default=16)
    judge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    judge.set_defaults(func=cmd_judge)

    score = subparsers.add_parser("score", help="Compute benchmark metrics")
    score.add_argument("--manifest", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--judges")
    score.add_argument("--output-dir", required=True)
    score.set_defaults(func=cmd_score)

    suite = subparsers.add_parser("suite", help="Run the complete single-turn suite")
    suite.add_argument("--endpoint", required=True)
    suite.add_argument("--judge-endpoint")
    suite.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION_ROOT))
    suite.add_argument("--run-dir", required=True)
    suite.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    suite.add_argument("--limit", type=int)
    suite.add_argument(
        "--max-model-len",
        type=int,
        help="Resolved server context length, recorded and sanity-checked.",
    )
    suite.add_argument("--workers", type=int, default=8)
    suite.add_argument("--judge-workers", type=int, default=16)
    suite.add_argument("--benchmark-workers", type=int, default=1)
    suite.add_argument(
        "--stream-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Judge each completed prediction while remaining inference continues.",
    )
    suite.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    _add_generation_args(suite)
    suite.set_defaults(func=cmd_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
