from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ENV_ROOT.parent
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from environment.hle.protocol_sanity_check import _build_record, _load_cases  # noqa: E402
from environment.hle.adapter import HLEAdapter  # noqa: E402
from environment.run_memrl_benchmark import load_benchmark_config  # noqa: E402


def run_numeric_boundary_sanity(cases_path: Path) -> dict:
    config = load_benchmark_config("hle")
    adapter = HLEAdapter(config=config, traj_dir=None)
    cases = _load_cases(cases_path)

    rows = []
    for case in cases:
        record = _build_record(case)
        prediction = str(case.get("prediction", "")).strip()
        extracted_answer = adapter._extract_final_answer(prediction)
        answer_contract_family = adapter._answer_contract_family(record)
        reasoning_family = adapter._reasoning_family(record)
        failure_signal = adapter._judge_failure_signal(
            extracted_answer,
            prediction,
            answer_contract_family,
            judge_error="",
        )
        success = failure_signal == "none"
        grading = {
            "prediction": prediction,
            "extracted_answer": extracted_answer,
            "gold_answer": adapter._gold_answer(record),
            "answer_type": adapter._answer_type(record),
            "failure_signal": failure_signal,
            "judge_mode": "sanity_local_classifier",
        }
        reward_profile = adapter._reward_profile(
            record=record,
            success=success,
            failure_signal=failure_signal,
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
        )
        rows.append(
            {
                "id": case.get("id", ""),
                "expected_equivalence": case.get("expected_equivalence", ""),
                "success": bool(success),
                "failure_signal": grading.get("failure_signal", ""),
                "observed_boundary": reward_profile.get("observed_boundary", ""),
                "correction_rule": reward_profile.get("correction_rule", ""),
                "next_reasoning_move": reward_profile.get("next_reasoning_move", ""),
                "proof_obligation": reward_profile.get("proof_obligation", ""),
            }
        )

    return {
        "cases_file": str(cases_path),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run numeric boundary sanity checks for HLE.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("numeric_boundary_sanity_cases.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_numeric_boundary_sanity(args.cases)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"[saved] {args.output}")
        return
    print(payload)


if __name__ == "__main__":
    main()
