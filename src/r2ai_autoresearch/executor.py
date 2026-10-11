from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import signal
from pathlib import Path
from string import Formatter
from typing import Any

from .io import atomic_json, load_json


SECRET = re.compile(r"(?:sk-[A-Za-z0-9]{16,}|(?:api[_-]?)?(?:key|token|secret)\s*[=:])", re.IGNORECASE)


class StrictFormatter(Formatter):
    def get_value(self, key: str | int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if isinstance(key, int):
            raise ValueError("positional command templates are forbidden")
        if key not in kwargs:
            raise ValueError(f"missing executor context field: {key}")
        return kwargs[key]


def sha256_path(path: Path) -> str:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_path(child)))
    return digest.hexdigest()


def plan_invocation(registry: dict[str, Any], invocation: dict[str, Any]) -> dict[str, Any]:
    if registry.get("schema") != "r2ai.safety-autoresearch.executor-registry.v1":
        raise ValueError("unsupported executor registry schema")
    if invocation.get("schema") != "r2ai.safety-autoresearch.executor-invocation.v1":
        raise ValueError("unsupported executor invocation schema")
    executor_id = invocation["executor"]
    try:
        spec = registry["executors"][executor_id]
    except KeyError as exc:
        raise ValueError(f"unknown executor: {executor_id}") from exc
    if spec.get("enabled") is not True:
        raise ValueError(f"executor is not enabled: {executor_id}: {spec.get('status')}")
    context = {key: str(value) for key, value in invocation.get("context", {}).items()}
    missing = sorted(set(spec.get("required_context", [])) - set(context))
    if missing:
        raise ValueError(f"missing required executor context: {','.join(missing)}")
    if any(SECRET.search(value) for value in context.values()):
        raise ValueError("credentials may not appear in executor context")
    formatter = StrictFormatter()
    argv = [formatter.format(token, **context) for token in spec["argv"]]
    if not argv or any(not token for token in argv):
        raise ValueError("executor rendered an empty argv token")
    supplied_environment = {str(key): str(value) for key, value in invocation.get("environment", {}).items()}
    extra_environment = sorted(set(supplied_environment) - set(spec.get("allowed_environment", [])))
    if extra_environment:
        raise ValueError(f"environment key not allowlisted: {','.join(extra_environment)}")
    if any(SECRET.search(value) for value in supplied_environment.values()):
        raise ValueError("credentials may not appear in executor environment")
    for environment_key, context_key in spec.get("environment_bindings", {}).items():
        expected = context.get(context_key)
        actual = supplied_environment.get(environment_key)
        if expected is None or actual != expected:
            raise ValueError(f"environment binding mismatch: {environment_key} must equal context.{context_key}")
    inputs = {name: formatter.format(template, **context) for name, template in spec.get("inputs", {}).items()}
    outputs = {name: formatter.format(template, **context) for name, template in spec.get("outputs", {}).items()}
    timeout = int(spec.get("timeout_seconds", 86400))
    if timeout <= 0:
        raise ValueError("executor timeout must be positive")
    return {
        "schema": "r2ai.safety-autoresearch.executor-plan.v1",
        "executor": executor_id,
        "executor_version": spec["version"],
        "status": spec["status"],
        "argv": argv,
        "environment": supplied_environment,
        "inputs": inputs,
        "outputs": outputs,
        "working_directory": context.get("working_directory", context.get("legacy_root", ".")),
        "timeout_seconds": timeout,
    }


def execute_invocation(registry_path: Path, invocation_path: Path) -> dict[str, Any]:
    registry, invocation = load_json(registry_path), load_json(invocation_path)
    plan = plan_invocation(registry, invocation)
    run_root = Path(invocation["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    for name, raw in plan["inputs"].items():
        if not Path(raw).exists():
            raise FileNotFoundError(f"executor input missing: {name}: {raw}")
    receipt_path = run_root / "executor_receipt.json"
    log_path = run_root / "executor.log"
    if receipt_path.exists():
        existing = load_json(receipt_path)
        _validate_reusable_receipt(existing, plan, receipt_path)
        return existing
    environment = {**os.environ, **plan["environment"]}
    input_hashes = {name: sha256_path(Path(path)) for name, path in plan["inputs"].items()}
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            plan["argv"], cwd=plan["working_directory"], env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=plan["timeout_seconds"])
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode != 0:
        raise RuntimeError(f"legacy executor failed with exit code {returncode}; see {log_path}")
    if input_hashes != {name: sha256_path(Path(path)) for name, path in plan["inputs"].items()}:
        raise RuntimeError("executor inputs changed during execution")
    missing_outputs = [f"{name}:{path}" for name, path in plan["outputs"].items() if not Path(path).exists()]
    if missing_outputs:
        raise RuntimeError(f"legacy executor did not produce required outputs: {','.join(missing_outputs)}")
    receipt = {
        "schema": "r2ai.safety-autoresearch.executor-receipt.v1",
        "executor": plan["executor"],
        "executor_version": plan["executor_version"],
        "plan_sha256": _payload_sha(plan),
        "input_sha256": input_hashes,
        "output_sha256": {name: sha256_path(Path(path)) for name, path in plan["outputs"].items()},
        "returncode": returncode,
        "log": str(log_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _validate_reusable_receipt(receipt: dict[str, Any], plan: dict[str, Any], receipt_path: Path) -> None:
    """Prove that a completed invocation is still the same immutable result.

    Existence checks alone are insufficient: an input or output may have been
    replaced after the receipt was written.  A resume is idempotent only when
    the plan and every declared artifact still match their recorded hashes.
    """
    if receipt.get("schema") != "r2ai.safety-autoresearch.executor-receipt.v1":
        raise RuntimeError(f"unsupported executor receipt: {receipt_path}")
    if receipt.get("plan_sha256") != _payload_sha(plan):
        raise RuntimeError(f"refusing to reuse a non-matching executor receipt: {receipt_path}")
    if receipt.get("executor") != plan["executor"] or receipt.get("executor_version") != plan["executor_version"]:
        raise RuntimeError(f"executor identity changed since receipt creation: {receipt_path}")
    if receipt.get("returncode") != 0:
        raise RuntimeError(f"executor receipt is not successful: {receipt_path}")

    for artifact_kind in ("inputs", "outputs"):
        paths = plan[artifact_kind]
        recorded = receipt.get(f"{artifact_kind[:-1]}_sha256")
        if not isinstance(recorded, dict) or set(recorded) != set(paths):
            raise RuntimeError(f"executor receipt has an incomplete {artifact_kind} hash set: {receipt_path}")
        for name, raw_path in paths.items():
            path = Path(raw_path)
            if not path.exists():
                raise RuntimeError(f"executor {artifact_kind[:-1]} is missing after completion: {name}: {path}")
            if recorded[name] != sha256_path(path):
                raise RuntimeError(f"executor {artifact_kind[:-1]} changed after completion: {name}: {path}")


def _payload_sha(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
