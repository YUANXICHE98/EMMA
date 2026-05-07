from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})"),
]

MODEL_VISIBLE_KEYS = {
    "prompt",
    "memory_context",
    "raw_llm_output",
    "action",
    "obs",
    "state_repr",
    "final_state_repr",
    "e",
    "embedding_text",
}

GOLD_KEYS = {
    "gold_answer",
    "correct_answer",
    "answer",
    "label",
}


def _iter_json_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in {".json", ".jsonl"} else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def _load_json_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"__raw_line__": line})
        return records
    try:
        return [json.loads(path.read_text(encoding="utf-8", errors="ignore"))]
    except json.JSONDecodeError:
        return [{"__raw_text__": path.read_text(encoding="utf-8", errors="ignore")}]


def _walk(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk(item, f"{path}[{idx}]")


def _leaf_key(path: str) -> str:
    tail = path.rsplit(".", 1)[-1]
    if "[" in tail:
        tail = tail.split("[", 1)[0]
    return tail


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return ""


def _collect_gold_values(record: Any) -> set[str]:
    values: set[str] = set()
    for path, value in _walk(record):
        if _leaf_key(path) not in GOLD_KEYS:
            continue
        text = _stringify(value).strip()
        if text and len(text) >= 2:
            values.add(text)
    return values


def _scan_record(record: Any, *, min_gold_len: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    gold_values = _collect_gold_values(record)
    for path, value in _walk(record):
        text = _stringify(value)
        if not text:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    {
                        "type": "secret_pattern",
                        "path": path,
                        "sample": text[:160],
                    }
                )
        key = _leaf_key(path)
        if key not in MODEL_VISIBLE_KEYS:
            continue
        for gold in gold_values:
            if len(gold) < min_gold_len:
                continue
            if gold in text:
                findings.append(
                    {
                        "type": "gold_in_model_visible_field",
                        "path": path,
                        "gold": gold[:80],
                        "sample": text[:160],
                    }
                )
    return findings


def audit_path(root: Path, *, min_gold_len: int) -> dict[str, Any]:
    files = _iter_json_files(root)
    findings: list[dict[str, Any]] = []
    scanned_records = 0
    for file_path in files:
        records = _load_json_records(file_path)
        for idx, record in enumerate(records):
            scanned_records += 1
            for finding in _scan_record(record, min_gold_len=min_gold_len):
                finding["file"] = str(file_path)
                finding["record_index"] = idx
                findings.append(finding)
    return {
        "root": str(root),
        "files_scanned": len(files),
        "records_scanned": scanned_records,
        "findings": findings,
        "passed": not findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HLE result artifacts for open-source risks.")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("results"),
        help="Result directory or JSON/JSONL file to scan.",
    )
    parser.add_argument(
        "--min-gold-len",
        type=int,
        default=3,
        help="Minimum gold string length to test for exact leakage.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = audit_path(args.root, min_gold_len=args.min_gold_len)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
