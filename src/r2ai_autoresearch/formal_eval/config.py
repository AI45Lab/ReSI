from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


DEFAULT_EVALUATION_ROOT = Path("data/cache/benchmarks")
ASSET_ROOT = Path(__file__).resolve().parent / "assets"

BENCHMARKS = (
    "harmbench",
    "wildjailbreak",
    "wj_adv_benign",
    "h_cot",
    "xstest",
    "mmlu_pro",
    "gpqa_diamond",
    "ifeval",
)

SAFETY_BENCHMARKS = {"harmbench", "wildjailbreak", "h_cot"}
REFUSAL_BENCHMARKS = {"xstest", "wj_adv_benign"}
JUDGE_BENCHMARKS = SAFETY_BENCHMARKS | REFUSAL_BENCHMARKS
GPQA_DIAMOND_LENGTH_MULTIPLIER = 1


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_tokens: int = 4096
    seed: int = 42
    extra_body: dict[str, Any] = field(default_factory=dict)
    retry_max_tokens: int | None = None
    truncation_policy: str = "error"


def generation_for_benchmark(
    benchmark: str, base: GenerationConfig
) -> GenerationConfig:
    """Resolve benchmark-specific decoding limits from a common base config."""
    if base.max_tokens < 1:
        raise ValueError("base max_tokens must be a positive integer")
    multiplier = GPQA_DIAMOND_LENGTH_MULTIPLIER if benchmark == "gpqa_diamond" else 1
    return replace(base, max_tokens=base.max_tokens * multiplier)


def max_model_len_for_benchmarks(
    benchmarks: list[str] | tuple[str, ...], base_max_model_len: int
) -> int:
    """Return the server context needed by the selected benchmark set."""
    if base_max_model_len < 1:
        raise ValueError("base_max_model_len must be a positive integer")
    multiplier = (
        GPQA_DIAMOND_LENGTH_MULTIPLIER
        if "gpqa_diamond" in set(benchmarks)
        else 1
    )
    return base_max_model_len * multiplier


@dataclass(frozen=True)
class Endpoint:
    model: str
    transport: str = "http"
    base_url: str = ""
    api_key: str = "EMPTY"
    request_extra_body: dict[str, Any] = field(default_factory=dict)
    default_system_prompt_file: str | None = None
    source_path: str | None = None
    timeout_seconds: float = 3600.0
    max_retries: int = 2
    transient_retries: int = 3

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Endpoint timeout_seconds must be positive and finite")
        if any(type(v) is not int or not 0 <= v <= 12 for v in (self.max_retries, self.transient_retries)):
            raise ValueError("Endpoint retry limits must be integers from 0 to 12")
        if self.transport != "http":
            raise ValueError("The release supports HTTP endpoints only")

    @classmethod
    def from_file(cls, path: str | Path) -> "Endpoint":
        config_path = Path(path).expanduser().resolve()
        obj = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_mapping(obj, source_path=config_path)

    @classmethod
    def from_mapping(
        cls, obj: dict[str, Any], *, source_path: str | Path | None = None
    ) -> "Endpoint":
        resolved_source = Path(source_path).expanduser().resolve() if source_path else None
        transport = str(obj.get("transport", "http")).strip().lower()
        if transport != "http":
            raise ValueError(
                f"Unsupported endpoint transport {transport!r}; the release supports HTTP endpoints"
            )

        api_key = obj.get("api_key")
        if not api_key and obj.get("api_key_file"):
            key_path = Path(obj["api_key_file"]).expanduser()
            if not key_path.is_absolute() and resolved_source is not None:
                key_path = resolved_source.parent / key_path
            api_key = key_path.read_text(encoding="utf-8").strip()
        if not api_key and obj.get("api_key_env"):
            api_key = os.environ.get(obj["api_key_env"])
        if not api_key:
            api_key = "EMPTY"

        base_url = str(obj.get("base_url", "")).rstrip("/")
        if not base_url:
            raise ValueError("Endpoint base_url is required")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        extra = dict(obj.get("request_extra_body") or {})
        if "chat_template_kwargs" in obj:
            extra["chat_template_kwargs"] = obj["chat_template_kwargs"]

        default_system_prompt_file: str | None = None
        if obj.get("default_system_prompt_file"):
            prompt_path = Path(obj["default_system_prompt_file"]).expanduser()
            if not prompt_path.is_absolute() and resolved_source is not None:
                prompt_path = resolved_source.parent / prompt_path
            prompt_path = prompt_path.resolve()
            if not prompt_path.is_file():
                raise FileNotFoundError(
                    f"Default system prompt file not found: {prompt_path}"
                )
            default_system_prompt_file = str(prompt_path)
        return cls(
            model=str(obj["model"]),
            transport=transport,
            base_url=base_url,
            api_key=api_key,
            request_extra_body=extra,
            default_system_prompt_file=default_system_prompt_file,
            source_path=str(resolved_source) if resolved_source else None,
            timeout_seconds=float(obj.get("timeout_seconds", 3600)),
            max_retries=obj.get("max_retries", 2),
            transient_retries=obj.get("transient_retries", 3),
        )

    def public_dict(self) -> dict[str, Any]:
        result = {
            "transport": self.transport,
            "base_url": self.base_url,
            "model": self.model,
            "request_extra_body": self.request_extra_body,
            "default_system_prompt_file": self.default_system_prompt_file,
            "source_path": self.source_path,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "transient_retries": self.transient_retries,
        }
        return result


def dump_json(path: str | Path, obj: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, output)
