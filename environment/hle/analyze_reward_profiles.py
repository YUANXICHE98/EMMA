from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

ENV_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ENV_ROOT.parent
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _ensure_openai_stub() -> None:
    if "openai" in sys.modules:
        return
    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub


_ensure_openai_stub()

from environment.hle.adapter import HLEAdapter  # noqa: E402


def _legacy_stage_profile(failure_signal: str, success: bool) -> tuple[float, str]:
    return HLEAdapter._failure_stage_profile(failure_signal, success)


def _build_record(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    return {
        "subject": meta.get("subject", ""),
        "question": meta.get("question_only", "") or result.get("task_description", ""),
        "answer": meta.get("gold_answer", ""),
        "answer_type": meta.get("answer_type", "text"),
        "choices": meta.get("choices", []),
        "category": meta.get("subject", ""),
    }


def analyze_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    adapter = HLEAdapter({}, None)
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        record = _build_record(result)
        success = bool(result.get("success", False))
        failure_signal = str(result.get("final_failure_signal", "") or "none").strip() or "none"
        meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        reasoning_family = str(meta.get("reasoning_family", "")).strip()
        answer_contract_family = str(meta.get("answer_contract_family", "")).strip()
        current = HLEAdapter._reward_profile(
            record=record,
            success=success,
            failure_signal=failure_signal,
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
        )
        legacy_phi, _ = _legacy_stage_profile(failure_signal, success)
        rows.append(
            {
                "source_file": str(path),
                "task_id": result.get("task_id", ""),
                "subject": meta.get("subject", ""),
                "subject_family": current.get("subject_family", ""),
                "reasoning_family": reasoning_family,
                "answer_contract_family": answer_contract_family,
                "success": success,
                "failure_signal": failure_signal,
                "legacy_stage_phi": round(float(legacy_phi), 4),
                "structure_overlap": round(float(current.get("structure_overlap", 0.0)), 4),
                "new_phi": round(float(current.get("topology_potential", 0.0)), 4),
                "delta_phi": round(float(current.get("topology_potential", 0.0)) - float(legacy_phi), 4),
                "value_signal": round(float(current.get("value_signal", 0.0)), 4),
                "question": record.get("question", ""),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# HLE Reward Profile Replay",
        "",
        "This report replays existing HLE result files through the current reward-profile logic",
        "and compares the new structure-aware potential against the old stage-only failure phi.",
        "",
    ]
    if not rows:
        lines.append("No rows found.")
        return "\n".join(lines)

    lines.extend(
        [
            "| task_id | subject | reasoning_family | contract | success | failure_signal | legacy_phi | overlap | new_phi | delta |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['subject']} | {row['reasoning_family']} | "
            f"{row['answer_contract_family']} | {row['success']} | {row['failure_signal']} | "
            f"{row['legacy_stage_phi']:.4f} | {row['structure_overlap']:.4f} | {row['new_phi']:.4f} | {row['delta_phi']:+.4f} |"
        )

    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(
            f"- `{row['task_id']}`: `{row['failure_signal']}` on `{row['reasoning_family']}` / "
            f"`{row['answer_contract_family']}` changed phi from `{row['legacy_stage_phi']:.4f}` "
            f"to `{row['new_phi']:.4f}` (delta `{row['delta_phi']:+.4f}`)."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay HLE reward profiles from saved result files.")
    parser.add_argument("files", nargs="+", help="Path(s) to hle_full.json / hle_*.json summaries.")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for file_arg in args.files:
        path = Path(file_arg)
        rows.extend(analyze_file(path))

    markdown = render_markdown(rows)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"[saved] {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
