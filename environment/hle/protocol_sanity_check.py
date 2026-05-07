from __future__ import annotations

import argparse
import json
import os
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
    try:
        __import__("openai")
        return
    except ImportError:
        stub = types.ModuleType("openai")
        stub.OpenAI = object
        sys.modules["openai"] = stub


_ensure_openai_stub()

from environment.hle.adapter import HLEAdapter  # noqa: E402
from environment.run_memrl_benchmark import load_benchmark_config  # noqa: E402


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    elif isinstance(payload, list):
        cases = payload
    else:
        raise RuntimeError(f"Unsupported sanity-case payload at {path}")
    if not isinstance(cases, list):
        raise RuntimeError(f"Expected a list of sanity cases at {path}")
    return [dict(item) for item in cases if isinstance(item, dict)]


def _build_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "answer": case.get("gold_answer", ""),
        "answer_type": case.get("answer_type", "text"),
        "choices": case.get("choices", []),
        "subject": case.get("subject", ""),
        "category": case.get("subject", ""),
    }


def _score_case(
    adapter: HLEAdapter,
    record: dict[str, Any],
    prediction: str,
    judge_mode: str,
) -> dict[str, Any]:
    adapter.runner_cfg["judge_mode"] = judge_mode
    success, grading = adapter._grade_answer(record, prediction)
    return {
        "judge_mode": judge_mode,
        "success": bool(success),
        "extracted_answer": grading.get("extracted_answer", ""),
        "resolved_prediction": grading.get("resolved_prediction", ""),
        "failure_signal": grading.get("failure_signal", ""),
        "judge_error": grading.get("judge_error", ""),
        "judge_raw": grading.get("judge_raw", ""),
        "judge_model": grading.get("judge_model", ""),
    }


def run_sanity_check(
    *,
    benchmark: str,
    cases_path: Path,
    include_llm_judge: bool,
) -> dict[str, Any]:
    config = load_benchmark_config(benchmark)
    adapter = HLEAdapter(config=config, traj_dir=None)
    cases = _load_cases(cases_path)

    rows: list[dict[str, Any]] = []
    summary = {
        "total_cases": 0,
        "local_exact_match_pass": 0,
        "llm_judge_pass": 0,
        "local_false_negatives": 0,
        "llm_false_negatives": 0,
    }

    for case in cases:
        record = _build_record(case)
        prediction = str(case.get("prediction", "")).strip()
        expected_correct = bool(case.get("expected_correct", True))
        expected_equiv = str(case.get("expected_equivalence", "should_pass")).strip()

        row = {
            "id": case.get("id", ""),
            "subject": case.get("subject", ""),
            "answer_type": case.get("answer_type", "text"),
            "question": case.get("question", ""),
            "gold_answer": case.get("gold_answer", ""),
            "prediction": prediction,
            "expected_correct": expected_correct,
            "expected_equivalence": expected_equiv,
            "notes": case.get("notes", ""),
        }
        local_result = _score_case(adapter, record, prediction, "local_exact_match")
        row["local_exact_match"] = local_result
        summary["total_cases"] += 1
        if local_result["success"]:
            summary["local_exact_match_pass"] += 1
        if expected_correct and not local_result["success"]:
            summary["local_false_negatives"] += 1

        if include_llm_judge:
            llm_result = _score_case(adapter, record, prediction, "llm_judge")
            row["llm_judge"] = llm_result
            if llm_result["success"]:
                summary["llm_judge_pass"] += 1
            if expected_correct and not llm_result["success"]:
                summary["llm_false_negatives"] += 1

        rows.append(row)

    return {
        "benchmark": benchmark,
        "cases_file": str(cases_path),
        "judge_model": adapter._judge_model() if include_llm_judge else "",
        "include_llm_judge": include_llm_judge,
        "summary": summary,
        "results": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    include_llm_judge = bool(report.get("include_llm_judge"))

    lines = [
        "# HLE Protocol Sanity Check",
        "",
        f"- Cases file: `{report.get('cases_file', '')}`",
        f"- Local judge false negatives: `{summary.get('local_false_negatives', 0)}` / `{summary.get('total_cases', 0)}`",
    ]
    if include_llm_judge:
        lines.append(
            f"- LLM judge false negatives: `{summary.get('llm_false_negatives', 0)}` / `{summary.get('total_cases', 0)}`"
        )
        lines.append(f"- Judge model: `{report.get('judge_model', '')}`")

    lines.extend(
        [
            "",
            "| id | expected | local | local_extract | llm | llm_extract | notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in report.get("results", []):
        local = row.get("local_exact_match", {})
        llm = row.get("llm_judge", {}) if include_llm_judge else {}
        lines.append(
            f"| {row.get('id', '')} | {row.get('expected_equivalence', '')} | "
            f"{local.get('success', False)} | {str(local.get('extracted_answer', '')).replace('|', '/')} | "
            f"{llm.get('success', '') if include_llm_judge else ''} | "
            f"{str(llm.get('extracted_answer', '')).replace('|', '/') if include_llm_judge else ''} | "
            f"{str(row.get('notes', '')).replace('|', '/')} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a protocol sanity check for HLE judge behavior.")
    parser.add_argument(
        "--benchmark",
        default="hle",
        help="Benchmark config to load. Defaults to hle.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("protocol_sanity_cases.json"),
        help="Path to manually curated sanity cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the raw JSON report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output path for a markdown report.",
    )
    parser.add_argument(
        "--include-llm-judge",
        action="store_true",
        help="Also run llm_judge in parallel with local_exact_match.",
    )
    args = parser.parse_args()

    include_llm_judge = args.include_llm_judge and bool(
        os.environ.get("MEMRL_HLE_JUDGE_API_KEY")
        or os.environ.get("HLE_JUDGE_API_KEY")
        or os.environ.get("MEMRL_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    report = run_sanity_check(
        benchmark=args.benchmark,
        cases_path=args.cases,
        include_llm_judge=include_llm_judge,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {args.output_json}")
    markdown = render_markdown(report)
    if args.output_md:
        args.output_md.write_text(markdown, encoding="utf-8")
        print(f"[saved] {args.output_md}")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
