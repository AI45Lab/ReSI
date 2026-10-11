from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import Endpoint, GenerationConfig
from .io import append_jsonl, latest_records


@lru_cache(maxsize=8)
def _reasoning_system_content(prompt_file: str) -> list[dict[str, Any]]:
    """Load the official Ministral reasoning prompt in its published message format."""
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    begin_marker = "[THINK]"
    end_marker = "[/THINK]"
    begin = prompt.find(begin_marker)
    end = prompt.find(end_marker)
    if begin < 0 or end < begin:
        raise ValueError(
            f"Reasoning system prompt lacks {begin_marker}...{end_marker}: {prompt_file}"
        )
    return [
        {"type": "text", "text": prompt[:begin]},
        {
            "type": "thinking",
            "thinking": prompt[begin + len(begin_marker) : end],
            "closed": True,
        },
        {"type": "text", "text": prompt[end + len(end_marker) :]},
    ]


def _with_default_system_prompt(
    endpoint: Endpoint, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not endpoint.default_system_prompt_file:
        return messages
    content = [dict(part) for part in _reasoning_system_content(
        endpoint.default_system_prompt_file
    )]
    remaining = list(messages)
    if remaining and remaining[0].get("role") == "system":
        custom = remaining.pop(0).get("content")
        if custom:
            if not isinstance(custom, str):
                raise TypeError("Benchmark system prompt must be a string")
            content[-1]["text"] = str(content[-1].get("text", "")) + "\n\n" + custom
    return [{"role": "system", "content": content}, *remaining]


def _http_json(
    endpoint: Endpoint,
    route: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> dict[str, Any]:
    url = endpoint.base_url.rstrip("/") + "/" + route.lstrip("/")
    request = urllib.request.Request(
        url,
        method="GET" if body is None else "POST",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def _is_transient_request_error(exc: Exception) -> bool:
    """Return true only for failures that can safely retry the same request."""
    if isinstance(exc, (ConnectionError, TimeoutError, urllib.error.URLError)):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "connection error",
        "connection refused",
        "connection reset",
        "remote end closed",
        "server disconnected",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "too many requests",
        "rate limit",
    )
    if any(marker in message for marker in transient_markers):
        return True
    return re.search(r"\bhttp (?:429|5\d\d)\b", message) is not None


_CONTEXT_LENGTH_ERROR = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"requested\s+(?P<output>\d+)\s+output tokens.*?"
    r"prompt contains\s+(?P<at_least>at least\s+)?(?P<input>\d+)\s+input tokens",
    flags=re.IGNORECASE | re.DOTALL,
)


def _context_safe_max_tokens(exc: Exception, current_max_tokens: int) -> int | None:
    """Return a smaller output budget only for a server-reported context overflow.

    The configured max_tokens remains the evaluation cap.  This merely avoids an
    impossible HTTP 400 when a multi-turn prompt has consumed most of the model
    context window.  A small margin protects against tokenizer/accounting drift.
    """
    match = _CONTEXT_LENGTH_ERROR.search(f"{type(exc).__name__}: {exc}")
    if match is None:
        # Newer vLLM rejects an output cap larger than the entire context
        # before reporting prompt length. Reduce to that advertised cap;
        # a subsequent prompt-overflow response can narrow it further.
        cap = re.search(
            r"max_tokens=\d+ cannot be greater than max_model_len=(?:max_total_tokens=)?(\d+)",
            str(exc), flags=re.IGNORECASE,
        )
        if cap is None:
            return None
        safe_max_tokens = int(cap.group(1)) - 64
        return safe_max_tokens if 0 < safe_max_tokens < current_max_tokens else None
    context_tokens = int(match.group("context"))
    input_tokens = int(match.group("input"))
    if match.group("at_least"):
        # Some vLLM versions report only the lower bound implied by the
        # requested output budget (context - output + 1), not the true prompt
        # length.  Halving converges quickly without pretending that lower
        # bound is an exact token count.
        safe_max_tokens = current_max_tokens // 2
    else:
        safe_max_tokens = context_tokens - input_tokens - 64
    if safe_max_tokens < 1 or safe_max_tokens >= current_max_tokens:
        return None
    return safe_max_tokens


def _request_json(
    endpoint: Endpoint,
    route: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> dict[str, Any]:
    return _http_json(endpoint, route, body=body, timeout=timeout)


def check_endpoint(endpoint: Endpoint, timeout: float = 30) -> dict[str, Any]:
    result = _request_json(endpoint, "models", timeout=timeout)
    model_ids = [str(item.get("id")) for item in result.get("data", [])]
    return {"reachable": True, "configured_model": endpoint.model, "models": model_ids}


def chat_completion(
    endpoint: Endpoint,
    messages: list[dict[str, Any]],
    generation: GenerationConfig,
    *,
    timeout: float | None = None,
    retries: int | None = None,
    transient_retries: int | None = None,
    transient_max_delay: float = 30.0,
) -> dict[str, Any]:
    timeout = endpoint.timeout_seconds if timeout is None else timeout
    retries = endpoint.max_retries if retries is None else retries
    transient_retries = endpoint.transient_retries if transient_retries is None else transient_retries
    body: dict[str, Any] = {
        "model": endpoint.model,
        "messages": _with_default_system_prompt(endpoint, messages),
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
        "seed": generation.seed,
    }
    if generation.temperature is None:body.pop('temperature')
    if generation.top_p is not None:
        body["top_p"] = generation.top_p
    body.update(endpoint.request_extra_body)
    body.update(generation.extra_body)
    configured_max_tokens = int(body["max_tokens"])
    context_window_adjustments: list[dict[str, int]] = []

    last_error: Exception | None = None
    max_retries = max(retries, transient_retries)
    for attempt in range(max_retries + 1):
        try:
            raw = _request_json(endpoint, "chat/completions", body=body, timeout=timeout)
            choice = raw["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content is None:
                content = ""
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning")
            from ..research.responses import split_response
            parsed=split_response(str(content))
            if not reasoning_content:reasoning_content=parsed['reasoning'] or None
            return {
                "response": parsed['final_answer'],
                "raw_response": str(content),
                "final_complete": parsed['final_complete'],
                "finish_reason": choice.get("finish_reason"),
                "usage": raw.get("usage"),
                "response_id": raw.get("id"),
                "reasoning_content": reasoning_content,
                "requested_max_tokens": configured_max_tokens,
                "effective_max_tokens": int(body["max_tokens"]),
                "context_window_adjusted": bool(context_window_adjustments),
                "context_window_adjustments": context_window_adjustments,
            }
        except Exception as exc:
            last_error = exc
            current_max_tokens = int(body["max_tokens"])
            safe_max_tokens = _context_safe_max_tokens(exc, current_max_tokens)
            if safe_max_tokens is not None:
                context_window_adjustments.append(
                    {
                        "from_max_tokens": current_max_tokens,
                        "to_max_tokens": safe_max_tokens,
                    }
                )
                body["max_tokens"] = safe_max_tokens
                continue
            retry_limit = transient_retries if _is_transient_request_error(exc) else retries
            if attempt >= retry_limit:
                break
            delay = min(transient_max_delay, (2**attempt) + random.random())
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def run_parallel_jsonl(
    items: Iterable[dict[str, Any]],
    output_path: str | Path,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    workers: int,
    resume: bool,
    completion_validator: Callable[[dict[str, Any]], bool] | None = None,
    success_callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = latest_records(output, require_response=True) if resume else {}
    if completion_validator is not None:
        completed = {
            item_id: row
            for item_id, row in completed.items()
            if completion_validator(row)
        }
    pending = [row for row in items if str(row["id"]) not in completed]
    mode = "a" if resume else "w"
    counts = {"submitted": len(pending), "succeeded": 0, "failed": 0, "skipped": len(completed)}
    if not pending:
        return counts

    with output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(worker, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                base = {
                    "id": row["id"],
                    "benchmark": row.get("benchmark"),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    result = future.result()
                    base.update(result)
                    counts["failed" if "error" in result else "succeeded"] += 1
                except Exception as exc:
                    base["error"] = f"{type(exc).__name__}: {exc}"
                    counts["failed"] += 1
                append_jsonl(handle, base)
                if "error" not in base and success_callback is not None:
                    success_callback(row, base)
    return counts


def generate_manifest(
    manifest: list[dict[str, Any]],
    endpoint: Endpoint,
    output_path: str | Path,
    generation: GenerationConfig,
    *,
    workers: int,
    resume: bool,
    success_callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    benchmarks = {str(row.get("benchmark")) for row in manifest}
    gpqa_policy_required = benchmarks == {"gpqa_diamond"}
    requested_generation = {
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "max_tokens": generation.max_tokens,
        "seed": generation.seed,
        "extra_body": generation.extra_body,
        "retry_max_tokens": generation.retry_max_tokens,
        "truncation_policy": generation.truncation_policy,
    }

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        messages = []
        if row.get("system_prompt"):
            messages.append({"role": "system", "content": row["system_prompt"]})
        messages.append({"role": "user", "content": row["prompt"]})
        result = chat_completion(endpoint, messages, generation)
        if result.get('finish_reason')=='length' and generation.retry_max_tokens is not None:
            if generation.retry_max_tokens<=generation.max_tokens:raise ValueError('Retry budget must exceed initial budget')
            first={k:result.get(k) for k in ('response','finish_reason','usage','effective_max_tokens')}
            result=chat_completion(endpoint,messages,replace(generation,max_tokens=generation.retry_max_tokens,retry_max_tokens=None))
            result['extended_budget_retry']={'initial':first,'retry_max_tokens':generation.retry_max_tokens}
        if not result.get("response", "").strip() or result.get("final_complete") is False:
            result["error"]="Missing complete final answer"
        elif result.get("finish_reason")=="length" and generation.truncation_policy=="error":
            result["error"]="Response still truncated after configured output-budget retry"
        result["requested_generation"] = requested_generation
        return result

    def compatible_existing(row: dict[str, Any]) -> bool:
        existing = row.get("requested_generation")
        if existing is None:
            return False
        return existing == requested_generation

    return run_parallel_jsonl(
        manifest,
        output_path,
        worker,
        workers=workers,
        resume=resume,
        completion_validator=compatible_existing,
        success_callback=success_callback,
    )
