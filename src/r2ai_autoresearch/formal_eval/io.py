from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def append_jsonl(handle, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row)
            count += 1
    return count


def latest_records(path: str | Path, *, require_response: bool = False) -> dict[str, dict]:
    output: dict[str, dict] = {}
    file_path = Path(path)
    if not file_path.exists():
        return output
    for row in read_jsonl(file_path):
        item_id = str(row.get("id", ""))
        if not item_id:
            continue
        if require_response and (row.get("error") or "response" not in row):
            continue
        output[item_id] = row
    return output

