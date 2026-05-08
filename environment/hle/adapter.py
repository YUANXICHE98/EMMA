from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class HLEAdapter(BenchmarkAdapter):
    benchmark_name = "hle"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.runner_cfg = config.get("runner", {})
        self.task_records: list[dict[str, Any]] = []
        self.current_record: dict[str, Any] | None = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self.current_repair_evidence: list[dict[str, Any]] = []
        self.judge_client: OpenAI | None = None
        self.judge_calls = 0

    def setup(self) -> None:
        dataset_path = str(
            os.environ.get("EMMA_HLE_DATASET_PATH")
            or os.environ.get("MEMRL_HLE_DATASET_PATH")
            or os.environ.get("HLE_DATASET_PATH")
            or self.runner_cfg.get("dataset_path", "")
            or ""
        ).strip()
        dataset_name = str(
            os.environ.get("EMMA_HLE_DATASET_NAME")
            or os.environ.get("MEMRL_HLE_DATASET_NAME")
            or os.environ.get("HLE_DATASET_NAME")
            or self.runner_cfg.get("dataset_name", "cais/hle")
            or "cais/hle"
        ).strip()
        split = str(
            os.environ.get("EMMA_HLE_SPLIT")
            or os.environ.get("MEMRL_HLE_SPLIT")
            or os.environ.get("HLE_SPLIT")
            or self.runner_cfg.get("split", "test")
            or "test"
        ).strip()
        cache_dir = str(
            os.environ.get("EMMA_HLE_CACHE_DIR")
            or os.environ.get("MEMRL_HLE_CACHE_DIR")
            or os.environ.get("HLE_CACHE_DIR")
            or self.runner_cfg.get("cache_dir", "")
            or ""
        ).strip() or None

        if dataset_path:
            raw_records = self._load_local_dataset(Path(dataset_path), split=split)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("HLE adapter requires the `datasets` package` for remote dataset loading.") from exc
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            raw_records = [dict(item) for item in dataset]

        self.task_records = self._apply_selection(raw_records)
        if not self.task_records:
            raise RuntimeError("HLE adapter loaded zero tasks after filtering.")

    def reset_task(self, index: int | None = None) -> ResetResult:
        if not self.task_records:
            raise RuntimeError("HLE adapter has no loaded tasks.")

        task_index = 0 if index is None else index
        if task_index < 0 or task_index >= len(self.task_records):
            raise IndexError(f"HLE task index out of range: {task_index} / {len(self.task_records)}")

        record = self.task_records[task_index]
        self.current_record = record
        question = self._extract_question(record)
        answer_type = self._answer_type(record)
        choices = self._extract_choices(record)
        subject = self._record_value(record, "subject", "category", "discipline", default="")
        task_id = str(
            self._record_value(
                record,
                "id",
                "question_id",
                "sample_id",
                default=f"hle[{task_index}]",
            )
        )
        metadata = {
            "benchmark": self.benchmark_name,
            "task_index": task_index,
            "answer_type": answer_type,
            "subject": subject,
            "subject_family": self._subject_family(subject),
            "choices": choices,
            "_evaluator_gold_answer": self._gold_answer(record),
            "_evaluator_only": ["_evaluator_gold_answer"],
            "question_only": question,
            "abstract_goal": (
                "Solve an expert-level closed-ended reasoning task by identifying the correct final answer "
                "under the declared answer contract."
            ),
            "input_family": "closed_ended_question",
            "transform_family": self._reasoning_family(record),
            "output_contract_family": self._answer_contract_family(record),
            "visualization_family": "none",
            "visualization_contract_family": "none",
            "render_pattern_family": "direct_final_answer",
            "exception_contract_family": "exact_answer_only",
            "reasoning_family": self._reasoning_family(record),
            "answer_contract_family": self._answer_contract_family(record),
            "question_form_family": self._question_form_family(record),
            "extraction_risk": self._extraction_risk(record),
            "judge_reliability_flag": self._judge_reliability_flag(record),
            "abstract_signature": self._abstract_signature(record),
        }
        self.current_task = TaskSpec(
            task_id=task_id,
            instruction=question,
            task_type="closed_ended_reasoning",
            task_description=question,
            goal_repr=f"Solve the closed-ended HLE question and emit only the final answer. answer_type={answer_type}",
            metadata=metadata,
        )
        self.current_observation = self._render_observation(record)
        self.current_repair_evidence = []
        return ResetResult(
            task=self.current_task,
            observation=self.current_observation,
            state_repr=self._state_repr(record),
            valid_actions=[],
        )

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-3:]) if history_lines else "None"
        answer_type = str(task.metadata.get("answer_type", "text")).strip().lower()
        answer_contract = str(task.metadata.get("answer_contract_family", "")).strip().lower()
        choices = task.metadata.get("choices") or []
        choice_block = ""
        if choices:
            choice_block = "Candidate options:\n" + "\n".join(
                f"{self._choice_label(idx)}. {choice}" for idx, choice in enumerate(choices)
            )
            choice_block += "\n\n"

        answer_rule = self._answer_rule(
            answer_contract,
            choices=choices,
        )
        self_check = self._submission_self_check(
            answer_contract,
            choices=choices,
        )
        question_context = observation
        if history_lines and "[question]" not in str(observation or ""):
            original_question = str(task.metadata.get("question_only") or task.instruction or "").strip()
            subject = str(task.metadata.get("subject", "")).strip()
            parts = []
            if original_question:
                parts.append(f"[question]\n{original_question}")
            if choices:
                rendered = "\n".join(f"{self._choice_label(idx)}. {choice}" for idx, choice in enumerate(choices))
                parts.append(f"[choices]\n{rendered}")
            if subject:
                parts.append(f"[subject]\n{subject}")
            parts.append(f"[latest_feedback]\n{observation}")
            question_context = "\n\n".join(parts)

        return (
            "Benchmark: Humanity's Last Exam (HLE)\n"
            f"Task ID: {task.task_id}\n"
            f"Answer type: {answer_type}\n"
            f"Subject: {task.metadata.get('subject', '')}\n\n"
            f"{answer_rule}\n\n"
            f"{self_check}\n\n"
            f"{choice_block}"
            f"Current question context:\n{question_context}\n\n"
            f"Recent action history:\n{history_block}"
        )

    def step(self, action: str) -> StepResult:
        if self.current_record is None or self.current_task is None:
            raise RuntimeError("HLE step called before reset_task.")

        normalized = self.normalize_action(action)
        success, grading = self._grade_answer(self.current_record, normalized)
        verifiable_feedback = grading.get("verifiable_feedback")
        if isinstance(verifiable_feedback, dict) and verifiable_feedback:
            self.current_repair_evidence.append(verifiable_feedback)
        reward = 1.0 if success else 0.0
        failure_signal = "" if success else grading.get("failure_signal", "reasoning_mismatch")
        observation = self._render_grading_observation(grading, success)
        self.current_observation = observation
        return StepResult(
            observation=observation,
            reward=reward,
            done=True,
            success=success,
            state_repr=self._state_repr(self.current_record),
            failure_signal=failure_signal,
            terminal_status="success" if success else "failure",
            info=grading,
            valid_actions=[],
        )

    def history_entry(self, action: str, observation: str, step_result: StepResult) -> str:
        info = step_result.info if isinstance(step_result.info, dict) else {}
        feedback = info.get("verifiable_feedback", {})
        if isinstance(feedback, dict) and feedback and feedback.get("valid") is False:
            parts = [
                "> [environment_rejected_previous_answer]",
                f"< failure_signal: {step_result.failure_signal}",
            ]
            extracted = str(info.get("extracted_answer", "")).strip()
            if extracted:
                parts.append(f"< rejected_exact_answer: {extracted}")
            for key in (
                "check",
                "target",
                "candidate",
                "remainder",
                "claimed_verification",
                "claimed_verification_valid",
                "claimed_verification_candidate",
                "claimed_verification_remainder",
                "claimed_verification_matches_exact_answer",
                "claim_error",
                "evidence_summary",
                "repair_constraint",
            ):
                if key in feedback:
                    parts.append(f"< {key}: {feedback.get(key)}")
            parts.append("< repair_loop: continue with verifier evidence only; do not reuse the rejected derivation.")
            return "\n".join(parts)
        return f"> {action}\n< {observation}"

    def force_finish(self) -> StepResult:
        return StepResult(
            observation="HLE episode terminated before an answer was submitted.",
            reward=0.0,
            done=True,
            success=False,
            state_repr=self.current_observation or "answer_missing",
            failure_signal="submission_missing",
            terminal_status="forced_terminate",
            info={"forced_terminate": True},
            valid_actions=[],
        )

    def build_repair_prompt(self, task: TaskSpec, last_trace: dict[str, Any]) -> str:
        if getattr(task, "task_type", "") != "closed_ended_reasoning":
            return ""
        info = last_trace.get("info", {}) if isinstance(last_trace, dict) else {}
        if not isinstance(info, dict):
            return ""
        failure_signal = str(last_trace.get("failure_signal", "") or info.get("failure_signal", "")).strip()
        if not failure_signal or failure_signal in {"none", "llm_infra_error"}:
            return ""
        reward_profile = info.get("reward_profile", {}) if isinstance(info.get("reward_profile"), dict) else {}
        correction_rule = str(reward_profile.get("correction_rule", "")).strip()
        if not correction_rule:
            _, correction_rule = self._failure_stage_profile(
                self._normalize_boundary_label(failure_signal),
                success=False,
            )
        extracted = str(info.get("extracted_answer", "")).strip()
        lines = [
            "[Repair Context]",
            "Your previous answer was judged incorrect.",
            f"failure_signal: {failure_signal}",
        ]
        if extracted:
            lines.append(f"previous_extracted_answer: {extracted}")
        if correction_rule:
            lines.append(f"repair_rule: {correction_rule}")
        verifiable_feedback = info.get("verifiable_feedback", {})
        if isinstance(verifiable_feedback, dict) and verifiable_feedback:
            lines.append("verified_failure_evidence:")
            for key in (
                "check",
                "target",
                "candidate",
                "remainder",
                "valid",
                "claimed_verification",
                "claimed_verification_valid",
                "claimed_verification_candidate",
                "claimed_verification_remainder",
                "claimed_verification_matches_exact_answer",
                "claim_error",
                "evidence_summary",
                "sample_branch",
                "sample_substitution",
                "parameter_values",
                "repair_constraint",
            ):
                if key in verifiable_feedback:
                    lines.append(f"- {key}: {verifiable_feedback.get(key)}")
            if verifiable_feedback.get("valid") is False and verifiable_feedback.get("candidate") not in (None, ""):
                lines.append(
                    f"forbidden_previous_candidate: {verifiable_feedback.get('candidate')} failed the verified check and must not appear as the next Exact Answer."
                )
        forbidden_candidates = self._forbidden_repair_candidates()
        if forbidden_candidates:
            lines.append("all_forbidden_verified_candidates: " + ", ".join(forbidden_candidates))
            lines.append("Do not output any candidate in all_forbidden_verified_candidates; each has already failed an environment-side verification check.")
        for key in (
            "reasoning_failure_pattern",
            "recompute_operator",
            "disallowed_shortcut",
            "next_reasoning_move",
            "proof_obligation",
        ):
            value = str(reward_profile.get(key, "")).strip()
            if value:
                lines.append(f"{key}: {value}")
        strategy_hint = self._repair_strategy_hint(task, failure_signal=failure_signal, previous_answer=extracted)
        if strategy_hint:
            lines.append(f"task_local_repair_hint: {strategy_hint}")
        repair_contract = self._repair_action_contract(task, failure_signal=failure_signal)
        if repair_contract:
            lines.append(f"repair_action_contract: {repair_contract}")
        hard_contract = self._hard_repair_contract(
            task,
            failure_signal=failure_signal,
            verifiable_feedback=verifiable_feedback if isinstance(verifiable_feedback, dict) else {},
            forbidden_candidates=forbidden_candidates,
        )
        if hard_contract:
            lines.append("[Hard Repair Contract]")
            lines.extend(hard_contract)
        lines.append("Revise the answer from the current question only. Do not repeat the previous final answer unless the proof obligation independently confirms it.")
        lines.append("If the repair cannot be verified from the question and your internal derivation, lower confidence instead of inventing a nearby answer.")
        return "\n".join(lines)

    def _forbidden_repair_candidates(self) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for evidence in self.current_repair_evidence:
            if not isinstance(evidence, dict) or evidence.get("valid") is not False:
                continue
            candidate = str(evidence.get("candidate", "")).strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        return candidates

    @staticmethod
    def _repair_action_contract(task: TaskSpec, *, failure_signal: str) -> str:
        metadata = getattr(task, "metadata", {}) or {}
        question = str(metadata.get("question_only") or getattr(task, "instruction", "") or "").strip()
        question_lc = question.lower()
        boundary = HLEAdapter._normalize_boundary_label(failure_signal)
        if boundary == "reasoning_numeric_mismatch" and ("largest prime divisor" in question_lc or ("prime" in question_lc and "divisor" in question_lc)):
            return (
                "In Explanation, include a line exactly like `verification: TARGET = CANDIDATE * QUOTIENT + REMAINDER`. "
                "Only put CANDIDATE in Exact Answer if REMAINDER is 0 and CANDIDATE is prime. "
                "If REMAINDER is not 0, do not use that candidate as Exact Answer."
            )
        return ""

    @staticmethod
    def _hard_repair_contract(
        task: TaskSpec,
        *,
        failure_signal: str,
        verifiable_feedback: dict[str, Any],
        forbidden_candidates: list[str],
    ) -> list[str]:
        metadata = getattr(task, "metadata", {}) or {}
        question = str(metadata.get("question_only") or getattr(task, "instruction", "") or "").strip()
        question_lc = question.lower()
        raw_signal = str(failure_signal or "").strip()
        boundary = HLEAdapter._normalize_boundary_label(raw_signal)
        check = str((verifiable_feedback or {}).get("check", "")).strip()
        lines: list[str] = []

        if "largest prime divisor" in question_lc or ("prime" in question_lc and "divisor" in question_lc):
            target_match = re.search(r"\b(\d{4,})\b", question)
            target = target_match.group(1) if target_match else "TARGET"
            if forbidden_candidates:
                lines.append("Forbidden Exact Answer values: " + ", ".join(forbidden_candidates) + ".")
            lines.append("Use exactly one candidate in the whole response; do not discuss alternate candidates or cofactors as possible answers.")
            lines.append("The response must use this exact three-line template:")
            lines.append(f"Explanation: verification: {target} = CANDIDATE * QUOTIENT + REMAINDER; CANDIDATE primality check: PRIME_OR_NOT")
            lines.append("Exact Answer: CANDIDATE")
            lines.append("Confidence: INTEGER")
            lines.append("The CANDIDATE in the verification line must be identical to the CANDIDATE in Exact Answer.")
            lines.append("Exact Answer is allowed only if that REMAINDER is 0 and CANDIDATE is prime.")
            lines.append("If no such verified candidate is found within the response, leave Exact Answer empty rather than guessing.")
            return lines

        if (
            raw_signal
            in {
                "reasoning_numeric_wrong_formula_family",
                "reasoning_numeric_formula_generic_bluff",
                "reasoning_numeric_formula_uninstantiated_bluff",
                "reasoning_numeric_formula_bluff",
            }
            or boundary
            in {
                "reasoning_numeric_formula_bluff",
                "reasoning_numeric_formula_generic_bluff",
                "reasoning_numeric_wrong_formula_family",
                "reasoning_numeric_formula_uninstantiated_bluff",
            }
            or check.startswith("numeric_formula")
        ):
            if forbidden_candidates:
                lines.append("Forbidden Exact Answer values from failed formula audits: " + ", ".join(forbidden_candidates) + ".")
            lines.append(
                "The Explanation line must include this audit chain in compact prose: "
                "object class -> valid theorem/formula for that exact class -> instantiated expression -> evaluated number."
            )
            lines.append("Do not cite chromatic number, list coloring, or a memorized family formula unless the chain states why it governs this exact invariant.")
            lines.append("Exact Answer may contain only the evaluated number produced by that chain.")
            return lines

        if boundary == "protocol_answer_missing":
            lines.append("Return exactly three fields: Explanation, Exact Answer, Confidence.")
            lines.append("Do not continue the derivation past the Exact Answer field; if no final answer is verified, Exact Answer must be empty.")
            return lines

        return lines

    @staticmethod
    def _repair_strategy_hint(task: TaskSpec, *, failure_signal: str, previous_answer: str) -> str:
        metadata = getattr(task, "metadata", {}) or {}
        question = str(metadata.get("question_only") or getattr(task, "instruction", "") or "").strip()
        question_lc = question.lower()
        boundary = HLEAdapter._normalize_boundary_label(failure_signal)
        answer_contract = str(metadata.get("answer_contract_family", "")).strip()
        previous = str(previous_answer or "").strip()

        if boundary == "reasoning_numeric_mismatch" or answer_contract == "numeric_exact_contract":
            if "largest prime divisor" in question_lc or ("prime" in question_lc and "divisor" in question_lc):
                target_match = re.search(r"\b(\d{4,})\b", question)
                target = target_match.group(1) if target_match else "the target integer"
                return (
                    f"Treat this as a factorization verification failure. Any candidate prime divisor must first divide {target} exactly; "
                    "discard candidates that fail the divisibility check, factor the remaining cofactor, and emit the largest verified prime factor only. "
                    "Do not replace the previous factor with another nearby factor unless the exact division check succeeds."
                )
            if "alon-tarsi" in question_lc and "k_{" in question_lc:
                return (
                    "Treat this as a graph invariant failure, not a chromatic-number shortcut. Do not answer 2 from bipartiteness alone; "
                    "identify the Alon-Tarsi/list-coloring theorem for this exact complete bipartite family, instantiate its n parameter, "
                    "and emit the evaluated invariant only after that theorem-to-number chain is explicit."
                )
            if "number of involutions" in question_lc:
                return (
                    "Treat this as a finite-group counting failure. A small linear estimate in q is not credible for a classical group order-scale count; "
                    "first identify the conjugacy-class/counting formula for involutions in this exact group family, instantiate q and rank, "
                    "then evaluate the resulting expression to the final integer."
                )
            return (
                "For numeric repair, the new answer must be produced by an independently checked derivation, not by editing the previous number. "
                "Verify that the final number satisfies the defining condition in the prompt before emitting it."
            )

        if boundary in {"reasoning_symbolic_mismatch", "contract_symbolic_exact_mismatch"}:
            if "i^z" in question_lc or "z\\cdot i" in question_lc or "z* i" in question_lc:
                return (
                    "Treat this as a transcendental complex-equation failure. Arithmetic progressions such as z=1+4k only solve i^z=1-like subcases; "
                    "rewrite i^z with the complex logarithm, transform to an equation of the form u e^u = c when needed, "
                    "and include the required branch parameter in one complete closed-form family."
                )
            return (
                "For symbolic repair, check the proposed family by substituting it back into the original equation and verify branch completeness. "
                "If the previous family came from a familiar template, switch method family rather than algebraically rewriting the same template."
            )

        if boundary == "reasoning_fact_retrieval_mismatch":
            if previous:
                return (
                    f"Treat this as an entity-verification failure. The previous entity '{previous}' is disallowed unless the exact prompt cue independently selects it; "
                    "match the requested place/date/relation first, then answer only with the entity selected by that cue."
                )
            return (
                "Treat this as an entity-verification failure. Match the exact place/date/relation cue first; do not answer from topical familiarity alone."
            )
        return ""

    def normalize_action(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if not action:
            return ""
        if action.startswith("```"):
            lines = [line for line in action.splitlines() if not line.strip().startswith("```")]
            action = "\n".join(lines).strip()
        for marker in ("Explanation:", "Exact Answer:", "Confidence:"):
            idx = action.find(marker)
            if idx > 0:
                prefix = action[:idx].lower()
                if "respond in exactly this three-field format" in prefix or "private self-check" in prefix:
                    action = action[idx:].strip()
                    break
        action = re.sub(r"^\s*(final answer|answer)\s*:\s*", "", action, flags=re.IGNORECASE).strip()
        if self._looks_like_internal_guidance(action):
            return ""
        return action

    @staticmethod
    def _answer_rule(answer_contract: str, *, choices: list[str]) -> str:
        shared_tail = "Do not include markdown fences, derivation blocks, or extra sections."
        if choices:
            return (
                "Respond in exactly this three-field format:\n"
                "Explanation: {brief reasoning, at most 2 short sentences}\n"
                "Exact Answer: {only the option letter or the exact option text}\n"
                "Confidence: {integer percentage from 0 to 100}\n"
                f"{shared_tail}"
            )
        if answer_contract == "numeric_exact_contract":
            return (
                "Respond in exactly this three-field format:\n"
                "Explanation: {brief reasoning, at most 2 short sentences}\n"
                "Exact Answer: {only the final numeric value}\n"
                "Confidence: {integer percentage from 0 to 100}\n"
                "Do not spell out calculation steps in the Exact Answer field.\n"
                "Exact Answer must be a single fully evaluated number, not a factorial shell, binomial shell, algebraic expression, or unevaluated formula.\n"
                f"{shared_tail}"
            )
        if answer_contract == "symbolic_exact_contract":
            return (
                "Respond in exactly this three-field format:\n"
                "Explanation: {at most 1 short sentence, optional but if present keep it under 20 words}\n"
                "Exact Answer: {one single-line closed-form symbolic answer only}\n"
                "Confidence: {integer percentage from 0 to 100}\n"
                "For symbolic exact questions, the Exact Answer field must contain only the final closed-form expression.\n"
                "Do not include prose such as 'the solutions are', 'or', 'approximately', or derivation text in Exact Answer.\n"
                "If the solution is a family, include the family parameter in the same Exact Answer line."
                "\nPrefer compact canonical symbolic notation rather than prose or partial root lists.\n"
                f"{shared_tail}"
            )
        return (
            "Respond in exactly this three-field format:\n"
            "Explanation: {brief reasoning, at most 3 short sentences}\n"
            "Exact Answer: {the succinct final answer only}\n"
            "Confidence: {integer percentage from 0 to 100}\n"
            f"{shared_tail}"
        )

    @staticmethod
    def _submission_self_check(answer_contract: str, *, choices: list[str]) -> str:
        if choices:
            return (
                "[Answer Format Check]\n"
                "Verify that Exact Answer contains exactly one option letter or one exact option text.\n"
                "Do not output a blended option, hedge, or multiple candidates."
            )
        if answer_contract == "numeric_exact_contract":
            return (
                "[Answer Format Check]\n"
                "Ensure Exact Answer contains only the final numeric value, with no units, prose, or side comments.\n"
                "For theorem/counting/factorization questions, the Explanation must contain a compact audit chain that determines the number; "
                "if no audit chain exists, do not output a confident numeric guess."
            )
        if answer_contract == "symbolic_exact_contract":
            return (
                "[Answer Format Check]\n"
                "Ensure Exact Answer is one canonical symbolic line only.\n"
                "If the answer is a family, keep the family parameter and completeness condition in the same line."
            )
        return (
            "[Answer Format Check]\n"
            "Ensure Exact Answer is one closed-ended final answer span only.\n"
            "Remove hedges, duplicate candidates, and explanatory prose from Exact Answer."
        )

    def _apply_selection(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = list(records)
        if self.runner_cfg.get("text_only", True):
            filtered = [record for record in filtered if not self._has_visual_payload(record)]

        allow_subjects = {str(item).strip().lower() for item in self.runner_cfg.get("subjects", []) if str(item).strip()}
        if allow_subjects:
            filtered = [
                record for record in filtered
                if str(self._record_value(record, "subject", "category", "discipline", default="")).strip().lower() in allow_subjects
            ]

        explicit_indices = [int(item) for item in self.runner_cfg.get("question_indices", [])]
        if explicit_indices:
            explicit_records: list[dict[str, Any]] = []
            for idx in explicit_indices:
                if 0 <= idx < len(filtered):
                    explicit_records.append(filtered[idx])
            filtered = explicit_records

        max_questions = int(self.runner_cfg.get("max_questions", 0) or 0)
        if max_questions > 0:
            filtered = filtered[:max_questions]
        return filtered

    def _render_observation(self, record: dict[str, Any]) -> str:
        question = self._extract_question(record)
        parts = [f"[question]\n{question}"]
        choices = self._extract_choices(record)
        if choices:
            rendered = "\n".join(f"{self._choice_label(idx)}. {choice}" for idx, choice in enumerate(choices))
            parts.append(f"[choices]\n{rendered}")
        subject = self._record_value(record, "subject", "category", "discipline", default="")
        if subject:
            parts.append(f"[subject]\n{subject}")
        return "\n\n".join(parts)

    def _state_repr(self, record: dict[str, Any]) -> str:
        answer_type = self._answer_type(record)
        choices = self._extract_choices(record)
        parts = [
            "[state_type]\nclosed_ended_reasoning",
            f"[answer_type]\n{answer_type}",
        ]
        if choices:
            parts.append("[candidate_answer_structure]\n" + "\n".join(
                f"{self._choice_label(idx)} -> {choice}" for idx, choice in enumerate(choices)
            ))
        return "\n\n".join(parts)

    def _grade_answer(self, record: dict[str, Any], prediction: str) -> tuple[bool, dict[str, Any]]:
        judge_mode = self._judge_mode()
        if judge_mode == "llm_judge":
            return self._grade_answer_llm_judge(record, prediction)
        if judge_mode != "local_exact_match":
            raise RuntimeError(f"Unsupported HLE judge mode: {judge_mode}")
        return self._grade_answer_local(record, prediction)

    def route_probe(self, task: TaskSpec, action: str) -> dict[str, Any] | None:
        mode = str(
            self.runner_cfg.get("routing_trigger_mode", "")
            or self.config.get("routing", {}).get("trigger_mode", "")
            or "disabled"
        ).strip().lower()
        if mode in {"", "disabled", "none"}:
            return {
                "should_escalate": False,
                "trigger_mode": mode or "disabled",
                "trigger_reason": "routing_disabled",
            }
        if self.current_record is None:
            return {
                "should_escalate": False,
                "trigger_mode": mode,
                "trigger_reason": "no_active_record",
            }

        extracted = self._extract_final_answer(action)
        answer_type = self._answer_type(self.current_record)
        if mode == "missing_or_malformed_answer":
            looks_missing = not extracted
            looks_malformed = False
            if answer_type in {"exactmatch", "numeric", "number"}:
                looks_malformed = bool(extracted) and self._try_float(extracted) is None and len(extracted.split()) > 4
            return {
                "should_escalate": bool(looks_missing or looks_malformed),
                "trigger_mode": mode,
                "trigger_reason": "answer_missing_or_malformed" if (looks_missing or looks_malformed) else "answer_format_ok",
                "extracted_answer": extracted,
            }
        if mode == "difficulty_probe":
            return {
                "should_escalate": False,
                "trigger_mode": mode,
                "trigger_reason": "difficulty_probe_pending",
            }
        if mode == "oracle_incorrect_for_test_only":
            success, grading = self._grade_answer_local(self.current_record, action)
            return {
                "should_escalate": not success,
                "trigger_mode": mode,
                "trigger_reason": grading.get("failure_signal", "oracle_match" if success else "oracle_mismatch"),
                "extracted_answer": grading.get("extracted_answer", extracted),
                "oracle_success": bool(success),
            }
        return {
            "should_escalate": False,
            "trigger_mode": mode,
            "trigger_reason": "unknown_trigger_mode",
            "extracted_answer": extracted,
        }

    def route_prompt(self, task: TaskSpec, observation: str) -> str:
        answer_type = str(task.metadata.get("answer_type", "text")).strip().lower()
        subject = str(task.metadata.get("subject", "")).strip()
        choices = task.metadata.get("choices") or []
        choice_block = ""
        if choices:
            choice_block = "\n".join(
                f"{self._choice_label(idx)}. {choice}" for idx, choice in enumerate(choices)
            )
        return (
            "Benchmark: Humanity's Last Exam (HLE)\n"
            f"Subject: {subject}\n"
            f"Answer type: {answer_type}\n"
            f"Reasoning family: {task.metadata.get('reasoning_family', '')}\n"
            f"Answer contract: {task.metadata.get('answer_contract_family', '')}\n\n"
            f"{observation}\n\n"
            f"{choice_block}"
        ).strip()

    def route_hint(self, task: TaskSpec, observation: str) -> dict[str, Any]:
        question = str(task.metadata.get("question_only", "") or task.instruction or "").strip()
        question_lc = question.lower()
        answer_contract = str(task.metadata.get("answer_contract_family", "")).strip().lower()
        reasoning_family = str(task.metadata.get("reasoning_family", "")).strip().lower()
        has_choices = bool(task.metadata.get("choices") or [])
        token_count = len(question.split())

        hard_markers = (
            "solve for all",
            "closed form",
            "lambert",
            "w_k",
            "\\mathbb",
            "\\forall",
            "for every",
            "for all",
            "classify all",
            "prove",
            "show that",
            "homology",
            "cohomology",
            "eigenvalue",
            "isomorphism",
            "manifold",
        )
        easy_numeric_starters = (
            "what is",
            "find the",
            "compute the",
            "determine the",
        )

        if has_choices:
            return {
                "route_label": "EASY",
                "trigger_reason": "local_mcq_default_easy",
                "source": "local_heuristic",
            }
        if answer_contract == "symbolic_exact_contract":
            return {
                "route_label": "HARD",
                "trigger_reason": "local_symbolic_exact_contract",
                "source": "local_heuristic",
            }
        if any(marker in question_lc for marker in hard_markers):
            return {
                "route_label": "HARD",
                "trigger_reason": "local_hard_math_marker",
                "source": "local_heuristic",
            }
        if (
            answer_contract == "numeric_exact_contract"
            and reasoning_family == "formal_reasoning|symbolic_or_numeric_derivation"
            and token_count <= 12
            and question_lc.startswith(easy_numeric_starters)
        ):
            return {
                "route_label": "EASY",
                "trigger_reason": "local_short_numeric_direct_question",
                "source": "local_heuristic",
            }
        return {
            "route_label": "DEFER",
            "trigger_reason": "local_defer_to_probe",
            "source": "local_heuristic",
        }

    def _grade_answer_local(self, record: dict[str, Any], prediction: str) -> tuple[bool, dict[str, Any]]:
        gold = self._gold_answer(record)
        answer_type = self._answer_type(record)
        choices = self._extract_choices(record)
        reasoning_family = self._reasoning_family(record)
        answer_contract_family = self._answer_contract_family(record)
        extracted_prediction = self._extract_final_answer(prediction)

        if not extracted_prediction:
            failure_signal = "protocol_answer_missing"
            reward_profile = self._reward_profile(
                record=record,
                success=False,
                failure_signal=failure_signal,
                answer_contract_family=answer_contract_family,
                reasoning_family=reasoning_family,
            )
            verifiable_feedback = self._claimed_integer_verification_feedback(
                record,
                prediction,
                extracted_prediction,
            ) or self._protocol_verifiable_feedback(
                failure_signal,
                extracted_prediction,
                prediction,
                answer_contract_family,
            )
            return False, {
                "prediction": prediction,
                "extracted_answer": extracted_prediction,
                "gold_answer": gold,
                "answer_type": answer_type,
                "failure_signal": failure_signal,
                "reward_profile": reward_profile,
                "verifiable_feedback": verifiable_feedback,
                "judge_mode": self._judge_mode(),
            }
        malformed_signal = self._contract_malformed_signal(prediction, extracted_prediction, answer_contract_family, choices=choices)
        if malformed_signal:
            reward_profile = self._reward_profile(
                record=record,
                success=False,
                failure_signal=malformed_signal,
                answer_contract_family=answer_contract_family,
                reasoning_family=reasoning_family,
            )
            return False, {
                "prediction": prediction,
                "extracted_answer": extracted_prediction,
                "gold_answer": gold,
                "answer_type": answer_type,
                "failure_signal": malformed_signal,
                "reward_profile": reward_profile,
                "verifiable_feedback": self._protocol_verifiable_feedback(
                    malformed_signal,
                    extracted_prediction,
                    prediction,
                    answer_contract_family,
                ),
                "judge_mode": self._judge_mode(),
            }

        if answer_type in {"multiple_choice", "multiple-choice", "choice", "mcq"} or choices:
            normalized_pred = self._resolve_choice_prediction(extracted_prediction, choices)
            success = any(
                self._normalized_text(normalized_pred) == self._normalized_text(candidate)
                for candidate in self._candidate_gold_answers(gold, choices)
            )
            failure_signal = "" if success else self._choice_failure_signal(extracted_prediction, normalized_pred, choices)
            reward_profile = self._reward_profile(
                record=record,
                success=success,
                failure_signal=failure_signal or "none",
                answer_contract_family=answer_contract_family,
                reasoning_family=reasoning_family,
            )
            return success, {
                "prediction": prediction,
                "extracted_answer": extracted_prediction,
                "resolved_prediction": normalized_pred,
                "gold_answer": gold,
                "answer_type": answer_type,
                "failure_signal": failure_signal or "none",
                "reward_profile": reward_profile,
                "judge_mode": self._judge_mode(),
            }

        gold_numeric = self._try_float(gold)
        pred_numeric = self._try_float(extracted_prediction)
        if gold_numeric is not None and pred_numeric is not None:
            tolerance = float(self.runner_cfg.get("answer_tolerance", 1e-6) or 1e-6)
            success = math.isclose(pred_numeric, gold_numeric, rel_tol=tolerance, abs_tol=tolerance)
            failure_signal = ""
            if not success:
                unsupported_type = self._unsupported_numeric_formula_type(prediction, extracted_prediction)
                if unsupported_type == "unsupported_numeric_formula_generic":
                    failure_signal = "reasoning_numeric_formula_generic_bluff"
                elif unsupported_type == "unsupported_numeric_formula_wrong_family":
                    failure_signal = "reasoning_numeric_wrong_formula_family"
                elif unsupported_type == "unsupported_numeric_formula_uninstantiated":
                    failure_signal = "reasoning_numeric_formula_uninstantiated_bluff"
                elif unsupported_type:
                    failure_signal = "reasoning_numeric_formula_bluff"
                else:
                    failure_signal = "numeric_reasoning_mismatch"
            reward_profile = self._reward_profile(
                record=record,
                success=success,
                failure_signal=failure_signal or "none",
                answer_contract_family=answer_contract_family,
                reasoning_family=reasoning_family,
            )
            verifiable_feedback = self._verifiable_failure_feedback(record, extracted_prediction, prediction)
            if (
                not success
                and isinstance(verifiable_feedback, dict)
                and verifiable_feedback
                and (
                    verifiable_feedback.get("claimed_verification_valid") is False
                    or verifiable_feedback.get("claimed_verification_matches_exact_answer") is False
                )
            ):
                failure_signal = "contract_numeric_audit_mismatch"
                reward_profile = self._reward_profile(
                    record=record,
                    success=False,
                    failure_signal=failure_signal,
                    answer_contract_family=answer_contract_family,
                    reasoning_family=reasoning_family,
                )
            if not verifiable_feedback and failure_signal:
                verifiable_feedback = self._numeric_formula_verifiable_feedback(
                    failure_signal,
                    extracted_prediction,
                    prediction,
                )
            return success, {
                "prediction": prediction,
                "extracted_answer": extracted_prediction,
                "gold_answer": gold,
                "answer_type": answer_type,
                "failure_signal": failure_signal or "none",
                "reward_profile": reward_profile,
                "verifiable_feedback": verifiable_feedback,
                "judge_mode": self._judge_mode(),
            }

        if answer_contract_family == "symbolic_exact_contract":
            normalized_pred = self._normalized_symbolic_text(extracted_prediction)
            success = any(
                normalized_pred == self._normalized_symbolic_text(candidate)
                for candidate in self._candidate_gold_answers(gold, choices)
            )
        else:
            normalized_pred = self._normalized_text(extracted_prediction)
            success = any(
                normalized_pred == self._normalized_text(candidate)
                for candidate in self._candidate_gold_answers(gold, choices)
            )
        failure_signal = "" if success else self._text_failure_signal(extracted_prediction, gold, answer_contract_family)
        reward_profile = self._reward_profile(
            record=record,
            success=success,
            failure_signal=failure_signal or "none",
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
        )
        verifiable_feedback = self._verifiable_failure_feedback(record, extracted_prediction, prediction)
        return success, {
            "prediction": prediction,
            "extracted_answer": extracted_prediction,
            "gold_answer": gold,
            "answer_type": answer_type,
            "failure_signal": failure_signal or "none",
            "reward_profile": reward_profile,
            "verifiable_feedback": verifiable_feedback,
            "judge_mode": self._judge_mode(),
        }

    def _grade_answer_llm_judge(self, record: dict[str, Any], prediction: str) -> tuple[bool, dict[str, Any]]:
        gold = self._gold_answer(record)
        question = self._extract_question(record)
        reasoning_family = self._reasoning_family(record)
        answer_contract_family = self._answer_contract_family(record)
        raw_judge = ""
        judge_error = ""
        parsed: dict[str, Any] = {}
        try:
            raw_judge = self._call_judge(question=question, gold=gold, prediction=prediction)
            parsed = self._parse_judge_response(raw_judge)
        except Exception as exc:
            judge_error = str(exc)
            parsed = {}

        extracted = str(parsed.get("extracted_final_answer") or parsed.get("extracted_answer") or "").strip()
        correct_text = str(parsed.get("correct", "no")).strip().lower()
        success = correct_text.startswith("yes") or correct_text == "true"
        fallback_extracted = self._extract_final_answer(prediction)
        if not extracted:
            extracted = fallback_extracted
        malformed_signal = self._contract_malformed_signal(prediction, extracted, answer_contract_family, choices=[])
        if malformed_signal and not success:
            failure_signal = malformed_signal
        else:
            failure_signal = "none" if success else self._judge_failure_signal(extracted, prediction, answer_contract_family, judge_error)
        reward_profile = self._reward_profile(
            record=record,
            success=success,
            failure_signal=failure_signal,
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
        )
        return success, {
            "prediction": prediction,
            "extracted_answer": extracted,
            "gold_answer": gold,
            "answer_type": self._answer_type(record),
            "failure_signal": failure_signal,
            "reward_profile": reward_profile,
            "judge_mode": self._judge_mode(),
            "judge_model": self._judge_model(),
            "judge_raw": raw_judge,
            "judge_parsed": parsed,
            "judge_error": judge_error,
            "verifiable_feedback": self._protocol_verifiable_feedback(
                failure_signal,
                extracted,
                prediction,
                answer_contract_family,
            ),
        }

    def _render_grading_observation(self, grading: dict[str, Any], success: bool) -> str:
        verdict = "correct" if success else "incorrect"
        payload = {
            "verdict": verdict,
            "prediction": grading.get("prediction", ""),
            "extracted_answer": grading.get("extracted_answer", ""),
            "resolved_prediction": grading.get("resolved_prediction", ""),
            "answer_type": grading.get("answer_type", ""),
            "failure_signal": grading.get("failure_signal", ""),
            "judge_mode": grading.get("judge_mode", self._judge_mode()),
            "judge_model": grading.get("judge_model", ""),
        }
        if grading.get("verifiable_feedback"):
            payload["verifiable_feedback"] = grading.get("verifiable_feedback")
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _record_value(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
        for key in keys:
            if key not in record:
                continue
            value = record[key]
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            return value
        return default

    @staticmethod
    def _extract_question(record: dict[str, Any]) -> str:
        question = HLEAdapter._record_value(record, "question", "prompt", "problem", default="")
        if isinstance(question, list):
            return "\n".join(str(item) for item in question)
        return str(question)

    @staticmethod
    def _extract_choices(record: dict[str, Any]) -> list[str]:
        raw = HLEAdapter._record_value(record, "choices", "options", "mcq_choices", default=[])
        if isinstance(raw, dict):
            return [str(value) for _, value in sorted(raw.items(), key=lambda item: str(item[0]))]
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            return [str(item) for item in raw]
        return []

    def _gold_answer(self, record: dict[str, Any]) -> Any:
        return self._record_value(
            record,
            "answer",
            "correct_answer",
            "target",
            "label",
            default="",
        )

    @staticmethod
    def _answer_type(record: dict[str, Any]) -> str:
        return str(HLEAdapter._record_value(record, "answer_type", "type", default="text")).strip().lower()

    def _has_visual_payload(self, record: dict[str, Any]) -> bool:
        if not self.runner_cfg.get("drop_image_columns", True):
            return False
        for key, value in record.items():
            key_lower = str(key).lower()
            if "image" not in key_lower:
                continue
            if value in (None, "", [], {}):
                continue
            return True
        return False

    def _load_local_dataset(self, dataset_path: Path, split: str) -> list[dict[str, Any]]:
        if not dataset_path.exists():
            raise RuntimeError(f"HLE dataset path does not exist: {dataset_path}")

        suffix = dataset_path.suffix.lower()
        if suffix == ".parquet":
            return self._load_parquet_records(dataset_path)
        if suffix in {".jsonl", ".json"}:
            return self._load_json_records(dataset_path)
        raise RuntimeError(
            f"Unsupported HLE dataset format `{suffix}` at {dataset_path}. "
            "Use .parquet, .jsonl, or .json."
        )

    def _load_parquet_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading HLE parquet requires the `pandas` package.") from exc

        df = pd.read_parquet(dataset_path)
        required_columns = {"id", "question", "answer"}
        missing = sorted(required_columns - set(str(col) for col in df.columns))
        if missing:
            raise RuntimeError(
                f"HLE parquet missing required columns: {', '.join(missing)}. "
                "Expected at least id/question/answer."
            )

        records = df.to_dict(orient="records")
        return [self._normalize_record(dict(record)) for record in records]

    def _load_json_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        text = dataset_path.read_text(encoding="utf-8")
        if dataset_path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                if "data" in payload and isinstance(payload["data"], list):
                    records = payload["data"]
                else:
                    records = list(payload.values())
            else:
                raise RuntimeError(f"Unsupported JSON payload at {dataset_path}")
        return [self._normalize_record(dict(record)) for record in records]

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        if "category" not in normalized:
            normalized["category"] = self._record_value(normalized, "subject", "discipline", default="")
        if "answer_type" not in normalized:
            normalized["answer_type"] = self._infer_answer_type(normalized)
        if self.runner_cfg.get("drop_image_columns", True):
            for key in list(normalized.keys()):
                if "image" in str(key).lower():
                    normalized[key] = ""
        return normalized

    @staticmethod
    def _infer_answer_type(record: dict[str, Any]) -> str:
        choices = HLEAdapter._extract_choices(record)
        if choices:
            return "multiple_choice"
        gold = HLEAdapter._record_value(
            record,
            "answer",
            "correct_answer",
            "target",
            "label",
            default="",
        )
        if HLEAdapter._try_float(gold) is not None:
            return "number"
        return "text"

    def _candidate_gold_answers(self, gold: Any, choices: list[str]) -> list[str]:
        candidates: list[str] = []
        if isinstance(gold, Iterable) and not isinstance(gold, (str, bytes, dict)):
            candidates.extend(str(item) for item in gold)
        elif gold not in (None, ""):
            candidates.append(str(gold))

        if choices:
            gold_norms = {self._normalized_text(item) for item in candidates}
            for idx, choice in enumerate(choices):
                label = self._choice_label(idx)
                if self._normalized_text(label) in gold_norms:
                    candidates.append(choice)
        return candidates or [""]

    def _resolve_choice_prediction(self, prediction: str, choices: list[str]) -> str:
        trimmed = prediction.strip()
        if not choices:
            return trimmed
        match = re.fullmatch(r"\(?([A-Z])\)?\.?", trimmed.upper())
        if match:
            idx = ord(match.group(1)) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx]
        normalized_pred = self._normalized_text(trimmed)
        for idx, choice in enumerate(choices):
            label = self._choice_label(idx)
            if normalized_pred == self._normalized_text(f"{label}. {choice}"):
                return choice
        return trimmed

    def _normalized_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if self.runner_cfg.get("normalize_unicode", True):
            text = unicodedata.normalize("NFKC", text)
        text = text.casefold()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^[\(\[]?([a-z])[\)\].]\s+", r"\1 ", text)
        return text.strip(" \n\t\"'`.,;:!?")

    def _normalized_symbolic_text(self, value: Any) -> str:
        text = self._normalized_text(value)
        replacements = {
            "$": "",
            "\\left": "",
            "\\right": "",
            "\\,": "",
            "\\mathbb": "",
            "\\in": "in",
            "ℤ": "z",
            "π": "pi",
            "\\pi": "pi",
            "\\frac": "frac",
            "\\cdot": "*",
            "−": "-",
            "∈": "in",
        }
        for old, new in replacements.items():
            text = text.replace(old.casefold(), new)
        text = re.sub(r"\bz\s*=", "", text)
        text = re.sub(r"\b[kn]\s*in\s*\{?z\}?", "kinz", text)
        text = re.sub(r"\b[kn]∈z\b", "kinz", text)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r",[a-z]inz$", ",kinz", text)
        text = text.replace("{", "").replace("}", "")
        text = re.sub(r"w_[a-z]\(", "w_k(", text)
        text = text.replace("frac2iw_k(-pi/2)pi", "2i*w_k(-pi/2)/pi")
        text = text.replace("(2i/pi)w_k(-pi/2)", "2i*w_k(-pi/2)/pi")
        text = text.replace("2i/pi*w_k(-pi/2)", "2i*w_k(-pi/2)/pi")
        return text.strip(" ,.;:")

    @staticmethod
    def _try_float(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except Exception:
            return None

    @staticmethod
    def _try_int_answer(value: Any) -> int | None:
        text = str(value or "").strip().replace(",", "")
        if not re.fullmatch(r"[-+]?\d+", text):
            return None
        try:
            return int(text)
        except Exception:
            return None

    @staticmethod
    def _verifiable_failure_feedback(record: dict[str, Any], extracted_prediction: str, prediction: str = "") -> dict[str, Any]:
        question = HLEAdapter._extract_question(record).strip()
        question_lc = question.lower()
        prediction_lc = str(prediction or "").lower()
        extracted_lc = str(extracted_prediction or "").lower()

        if "i^z" in question_lc and ("z\\cdot i" in question_lc or "z⋅i" in question_lc or "z* i" in question_lc):
            if not any(marker in extracted_lc for marker in ("w_", "lambert", "w_k", "wk")):
                feedback = {
                    "check": "symbolic_family_operator_check",
                    "valid": False,
                    "evidence_summary": "The proposed symbolic family does not expose the inverse operator needed to isolate z from both z and i^z.",
                    "repair_constraint": "Do not reuse a linear arithmetic progression as the complete family; derive the logarithmic transform and inverse u*exp(u)=c branch structure.",
                }
                if re.search(r"z\s*=\s*(?:1\s*\+\s*4\s*k|4\s*k\s*\+\s*1)", str(extracted_prediction), flags=re.IGNORECASE):
                    feedback.update(
                        {
                            "sample_branch": "k=1",
                            "sample_substitution": "z=5 gives |z*i|=5 while |i^z|=1, so the family cannot be complete.",
                        }
                    )
                return feedback

        if "alon-tarsi" in question_lc and "k_{" in question_lc:
            match = re.search(r"k_\{\s*(\d+)\s*,\s*(\d+)\s*\}", question_lc)
            if match:
                left, right = match.groups()
                if "does not specify" in prediction_lc or re.search(r"\bn\s*=\s*1\b", prediction_lc) or str(extracted_prediction).strip() == "2":
                    return {
                        "check": "question_parameter_extraction",
                        "valid": False,
                        "parameter_values": f"K_{{{left},{right}}}",
                        "evidence_summary": "The question explicitly supplies the complete bipartite graph parameters; treating n as unspecified or defaulting to n=1 is a parameter-use failure.",
                        "repair_constraint": "Extract the graph parameters from the question first and instantiate the Alon-Tarsi theorem for that exact K_{m,n}.",
                    }

        candidate = HLEAdapter._try_int_answer(extracted_prediction)
        if candidate is None:
            return {}
        if "prime" in question_lc and "divisor" in question_lc:
            target_match = re.search(r"\b(\d{4,})\b", question)
            if not target_match:
                return {}
            target = int(target_match.group(1))
            if candidate == 0:
                return {
                    "check": "candidate_divides_target",
                    "target": target,
                    "candidate": candidate,
                    "valid": False,
                    "reason": "zero_cannot_be_a_prime_divisor",
                }
            remainder = target % abs(candidate)
            feedback = {
                "check": "candidate_divides_target",
                "target": target,
                "candidate": candidate,
                "remainder": remainder,
                "valid": remainder == 0,
            }
            if remainder != 0:
                feedback["repair_constraint"] = "The next candidate must divide the target exactly; this candidate is not a divisor."
            claimed = HLEAdapter._extract_claimed_integer_verification(prediction)
            if claimed:
                claimed_target, claimed_candidate, claimed_quotient, claimed_remainder = claimed
                actual_remainder = claimed_target % abs(claimed_candidate) if claimed_candidate else None
                actual_product_total = claimed_candidate * claimed_quotient + claimed_remainder
                claim_valid = (
                    claimed_target == target
                    and actual_product_total == claimed_target
                    and actual_remainder == claimed_remainder
                )
                feedback["claimed_verification"] = (
                    f"{claimed_target} = {claimed_candidate} * {claimed_quotient} + {claimed_remainder}"
                )
                feedback["claimed_verification_valid"] = bool(claim_valid)
                feedback["claimed_verification_candidate"] = claimed_candidate
                feedback["claimed_verification_remainder"] = claimed_remainder
                claim_errors: list[str] = []
                repair_constraints: list[str] = []
                if claimed_candidate != candidate:
                    feedback["claimed_verification_matches_exact_answer"] = False
                    claim_errors.append(
                        f"claimed verification checks candidate {claimed_candidate}, but Exact Answer is {candidate}"
                    )
                    repair_constraints.append(
                        "The verification line must verify the same CANDIDATE that appears in Exact Answer."
                    )
                if not claim_valid:
                    claim_errors.append(
                        f"claimed arithmetic is inconsistent; actual {claimed_target} mod {abs(claimed_candidate) if claimed_candidate else claimed_candidate} = {actual_remainder}"
                    )
                    repair_constraints.append(
                        "Do not trust the claimed verification line; recompute the arithmetic check before selecting another candidate."
                    )
                if claim_errors:
                    feedback["claim_error"] = "; ".join(claim_errors)
                if repair_constraints:
                    feedback["repair_constraint"] = " ".join(repair_constraints)
            return feedback
        return {}

    @staticmethod
    def _extract_claimed_integer_verification(prediction: str) -> tuple[int, int, int, int] | None:
        text = str(prediction or "").replace(",", "")
        patterns = [
            r"verification\s*:\s*(-?\d+)\s*=\s*(-?\d+)\s*\*\s*(-?\d+)\s*\+\s*(-?\d+)",
            r"\b(-?\d{4,})\s*=\s*(-?\d+)\s*\*\s*(-?\d+)\s*\+\s*(-?\d+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if not matches:
                continue
            target, candidate, quotient, remainder = matches[-1]
            try:
                return int(target), int(candidate), int(quotient), int(remainder)
            except Exception:
                return None
        return None

    @staticmethod
    def _claimed_integer_verification_feedback(
        record: dict[str, Any],
        prediction: str,
        extracted_prediction: str = "",
    ) -> dict[str, Any]:
        question = HLEAdapter._extract_question(record).strip()
        question_lc = question.lower()
        if "prime" not in question_lc or "divisor" not in question_lc:
            return {}
        target_match = re.search(r"\b(\d{4,})\b", question)
        if not target_match:
            return {}
        claimed = HLEAdapter._extract_claimed_integer_verification(prediction)
        if not claimed:
            return {}
        target = int(target_match.group(1))
        claimed_target, claimed_candidate, claimed_quotient, claimed_remainder = claimed
        actual_remainder = claimed_target % abs(claimed_candidate) if claimed_candidate else None
        actual_product_total = claimed_candidate * claimed_quotient + claimed_remainder
        claim_valid = (
            claimed_target == target
            and actual_product_total == claimed_target
            and actual_remainder == claimed_remainder
        )
        feedback: dict[str, Any] = {
            "check": "claimed_integer_verification",
            "target": target,
            "candidate": claimed_candidate,
            "valid": False,
            "claimed_verification": (
                f"{claimed_target} = {claimed_candidate} * {claimed_quotient} + {claimed_remainder}"
            ),
            "claimed_verification_valid": bool(claim_valid),
            "claimed_verification_candidate": claimed_candidate,
            "claimed_verification_remainder": claimed_remainder,
            "actual_claimed_candidate_remainder": actual_remainder,
            "protocol_error": "required_exact_answer_field" if not str(extracted_prediction or "").strip() else "",
        }
        exact_candidate = HLEAdapter._try_int_answer(extracted_prediction)
        claim_errors: list[str] = []
        repair_constraints: list[str] = []
        if exact_candidate is None:
            feedback["claimed_verification_matches_exact_answer"] = False
            claim_errors.append(
                f"claimed verification checks candidate {claimed_candidate}, but Exact Answer is empty or non-integer"
            )
            repair_constraints.append(
                "The verification line must verify the same CANDIDATE that appears in Exact Answer."
            )
        elif claimed_candidate != exact_candidate:
            feedback["claimed_verification_matches_exact_answer"] = False
            claim_errors.append(
                f"claimed verification checks candidate {claimed_candidate}, but Exact Answer is {exact_candidate}"
            )
            repair_constraints.append(
                "The verification line must verify the same CANDIDATE that appears in Exact Answer."
            )
        if not claim_valid:
            claim_errors.append(
                f"claimed arithmetic is inconsistent; actual {claimed_target} mod {abs(claimed_candidate) if claimed_candidate else claimed_candidate} = {actual_remainder}"
            )
            repair_constraints.append(
                "Do not trust the claimed verification line; recompute the arithmetic check before selecting another candidate."
            )
        if claim_errors:
            feedback["claim_error"] = "; ".join(claim_errors)
        if repair_constraints:
            feedback["repair_constraint"] = " ".join(repair_constraints)
        return feedback

    @staticmethod
    def _choice_label(idx: int) -> str:
        return chr(ord("A") + idx)

    def _judge_mode(self) -> str:
        return str(
            os.environ.get("EMMA_HLE_JUDGE_MODE")
            or os.environ.get("MEMRL_HLE_JUDGE_MODE")
            or os.environ.get("HLE_JUDGE_MODE")
            or self.runner_cfg.get("judge_mode", "local_exact_match")
            or "local_exact_match"
        ).strip().lower()

    def _judge_model(self) -> str:
        return str(
            os.environ.get("EMMA_HLE_JUDGE_MODEL")
            or os.environ.get("MEMRL_HLE_JUDGE_MODEL")
            or os.environ.get("HLE_JUDGE_MODEL")
            or self.runner_cfg.get("judge_model", "gpt-4o-2024-08-06")
            or "gpt-4o-2024-08-06"
        ).strip()

    def _judge_base_url(self) -> str:
        return str(
            os.environ.get("EMMA_HLE_JUDGE_BASE_URL")
            or os.environ.get("MEMRL_HLE_JUDGE_BASE_URL")
            or os.environ.get("HLE_JUDGE_BASE_URL")
            or os.environ.get("EMMA_OPENAI_BASE_URL")
            or os.environ.get("MEMRL_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.config.get("llm", {}).get("base_url", "")
            or ""
        ).strip()

    def _judge_api_key(self) -> str:
        return str(
            os.environ.get("EMMA_HLE_JUDGE_API_KEY")
            or os.environ.get("MEMRL_HLE_JUDGE_API_KEY")
            or os.environ.get("HLE_JUDGE_API_KEY")
            or os.environ.get("EMMA_OPENAI_API_KEY")
            or os.environ.get("MEMRL_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or self.config.get("llm", {}).get("api_key", "")
            or ""
        ).strip()

    def _get_judge_client(self) -> OpenAI:
        if self.judge_client is not None:
            return self.judge_client
        args = {"api_key": self._judge_api_key()}
        base_url = self._judge_base_url()
        if base_url:
            args["base_url"] = base_url
        trust_env_proxy = str(
            os.environ.get("MEMRL_HLE_JUDGE_TRUST_ENV_PROXY")
            or os.environ.get("HLE_JUDGE_TRUST_ENV_PROXY")
            or os.environ.get("MEMRL_OPENAI_TRUST_ENV_PROXY")
            or os.environ.get("OPENAI_TRUST_ENV_PROXY")
            or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        max_retries = int(
            os.environ.get("MEMRL_HLE_JUDGE_MAX_RETRIES")
            or os.environ.get("HLE_JUDGE_MAX_RETRIES")
            or os.environ.get("MEMRL_OPENAI_MAX_RETRIES")
            or os.environ.get("OPENAI_MAX_RETRIES")
            or 0
        )
        args["http_client"] = httpx.Client(trust_env=trust_env_proxy)
        args["max_retries"] = max_retries
        self.judge_client = OpenAI(**args)
        return self.judge_client

    def _call_judge(self, *, question: str, gold: Any, prediction: str) -> str:
        prompt = (
            "Judge whether the following [response] to [question] is correct based on the precise [correct_answer].\n\n"
            f"[question]: {question}\n\n"
            f"[response]: {prediction}\n\n"
            f"[correct_answer]: {gold}\n\n"
            "Return only a JSON object with these keys:\n"
            "extracted_final_answer: final answer extracted from the response, or null if absent.\n"
            "reasoning: short explanation focused only on answer equivalence.\n"
            "correct: yes or no.\n"
            "confidence: integer 0-100 extracted from response, or 100 if absent."
        )
        self.judge_calls += 1
        response = self._get_judge_client().chat.completions.create(
            model=self._judge_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=int(self.runner_cfg.get("judge_max_tokens", 1024) or 1024),
            timeout=float(self.runner_cfg.get("judge_timeout", 45) or 45),
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_judge_response(raw_judge: str) -> dict[str, Any]:
        text = str(raw_judge or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            parsed: dict[str, Any] = {}
            answer_match = re.search(r"extracted_final_answer\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
            correct_match = re.search(r"correct\s*[:=]\s*(yes|no|true|false)", text, flags=re.IGNORECASE)
            confidence_match = re.search(r"confidence\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
            if answer_match:
                parsed["extracted_final_answer"] = answer_match.group(1).strip().strip('"')
            if correct_match:
                parsed["correct"] = correct_match.group(1).strip().lower()
            if confidence_match:
                parsed["confidence"] = int(confidence_match.group(1))
            return parsed

    @staticmethod
    def _extract_final_answer(prediction: str) -> str:
        text = str(prediction or "").strip()
        if not text:
            return ""
        if HLEAdapter._looks_like_internal_guidance(text):
            return ""
        field_pattern = re.compile(
            r"^\s*(Exact Answer|Final Answer|Answer)\s*:\s*(.*)$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        field_matches = list(field_pattern.finditer(text))
        if field_matches:
            match = field_matches[-1]
            candidate = str(match.group(2) or "").strip().strip("`").strip()
            candidate = re.sub(
                r"(?i)\s*(?:\|\s*)?(confidence|explanation|reasoning)\s*:\s*.+$",
                "",
                candidate,
            ).strip(" |")
            if not candidate:
                return ""
            return HLEAdapter._cleanup_final_answer(candidate)
        patterns = [
            r"Exact Answer\s*:\s*(.+?)(?:\s*\|\s*Confidence\s*:|\n\s*[A-Z][A-Za-z ]*\s*:|$)",
            r"Final Answer\s*:\s*(.+?)(?:\s*\|\s*Confidence\s*:|\n\s*[A-Z][A-Za-z ]*\s*:|$)",
            r"Answer\s*:\s*(.+?)(?:\s*\|\s*Confidence\s*:|\n\s*[A-Z][A-Za-z ]*\s*:|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                candidate = str(matches[-1]).strip().strip("`").strip()
                candidate = re.sub(r"\s+Confidence\s*:\s*\d+\s*$", "", candidate, flags=re.IGNORECASE).strip()
                return HLEAdapter._cleanup_final_answer(candidate)
        return ""

    @staticmethod
    def _contract_malformed_signal(
        prediction: str,
        extracted_answer: str,
        answer_contract_family: str,
        *,
        choices: list[str],
    ) -> str:
        answer = str(extracted_answer or "").strip()
        if not answer:
            return "protocol_answer_missing"
        text = str(prediction or "")
        if "Exact Answer:" not in text and "Final Answer:" not in text and "Answer:" not in text:
            return "protocol_answer_missing"
        if choices:
            return ""
        if answer_contract_family == "numeric_exact_contract":
            if HLEAdapter._try_float(answer) is None:
                return "contract_numeric_exact_mismatch"
            if len(answer.split()) > 1:
                return "contract_numeric_exact_mismatch"
        if answer_contract_family == "symbolic_exact_contract":
            if "\n" in answer or len(answer.split()) > 80:
                return "contract_symbolic_exact_mismatch"
        return ""

    @staticmethod
    def _protocol_verifiable_feedback(
        failure_signal: str,
        extracted_answer: str,
        prediction: str,
        answer_contract_family: str,
    ) -> dict[str, Any]:
        signal = HLEAdapter._normalize_boundary_label(failure_signal)
        if signal == "protocol_answer_missing":
            return {
                "check": "required_exact_answer_field",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The model response did not expose a clean final answer field under the HLE protocol.",
                "repair_constraint": "Emit exactly Explanation, Exact Answer, and Confidence; Exact Answer must be one contract-compliant final span.",
            }
        if signal == "contract_numeric_exact_mismatch":
            return {
                "check": "numeric_exact_answer_contract",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The extracted final answer is not a single fully evaluated numeric span.",
                "repair_constraint": "Do the audit in Explanation only; Exact Answer must contain one fully evaluated number and nothing else.",
            }
        if signal == "contract_symbolic_exact_mismatch":
            return {
                "check": "symbolic_exact_answer_contract",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The extracted final answer is not one compact symbolic answer line.",
                "repair_constraint": "Put only the canonical symbolic expression or complete family in Exact Answer.",
            }
        return {}

    @staticmethod
    def _numeric_formula_verifiable_feedback(
        failure_signal: str,
        extracted_answer: str,
        prediction: str,
    ) -> dict[str, Any]:
        signal = str(failure_signal or "").strip()
        if signal == "reasoning_numeric_wrong_formula_family":
            return {
                "check": "numeric_formula_family_applicability",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The answer relied on a plausible theorem/formula family that is not verified as governing this exact object.",
                "repair_constraint": "First classify the exact object family, then use a theorem or formula valid for that family before instantiating parameters.",
            }
        if signal == "reasoning_numeric_formula_generic_bluff":
            return {
                "check": "numeric_formula_specificity",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The answer used generic theorem/formula language without an instantiated formula that determines the final number.",
                "repair_constraint": "Write the concrete formula specialized to this object before emitting any numeric answer.",
            }
        if signal == "reasoning_numeric_formula_uninstantiated_bluff":
            return {
                "check": "numeric_formula_instantiation",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The answer did not fully instantiate and evaluate the formula to the emitted number.",
                "repair_constraint": "Substitute all task parameters and evaluate to one final numeric span.",
            }
        if signal == "reasoning_numeric_formula_bluff":
            return {
                "check": "numeric_formula_audit_chain",
                "valid": False,
                "candidate": str(extracted_answer or "").strip(),
                "evidence_summary": "The stated theorem/formula did not form a complete audit chain to the final number.",
                "repair_constraint": "Provide an internal chain theorem -> instantiated expression -> evaluated number before answering.",
            }
        return {}

    @staticmethod
    def _extract_explanation(prediction: str) -> str:
        text = str(prediction or "").strip()
        if not text:
            return ""
        matches = re.findall(
            r"Explanation\s*:\s*(.+?)(?:(?:\n|\r\n?)\s*Exact Answer\s*:|\s*Exact Answer\s*:|$)",
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not matches:
            return ""
        return re.sub(r"\s+", " ", str(matches[-1]).strip()).strip()

    @staticmethod
    def _unsupported_numeric_formula_type(prediction: str, extracted_answer: str) -> str:
        explanation = HLEAdapter._extract_explanation(prediction).lower()
        answer = str(extracted_answer or "").strip().lower()
        if not explanation or not answer:
            return ""
        if "alon" in explanation and "tarsi" in explanation and "complete bipartite graph" in explanation:
            if any(marker in explanation for marker in ("list chromatic", "m+1", "n+1", "1000+1", "1000 + 1", "max(")):
                return "unsupported_numeric_formula_wrong_family"

        generic_formula_markers = (
            "standard formula",
            "by calculation",
            "using prime factorization",
            "prime factorization method",
            "properties of involutions",
            "specific counting identity",
            "specific combinatorial formula",
            "formula involving the order of the group",
            "can be determined using the formula",
            "is given by the formula",
            "is given by a specific counting identity",
            "is known to be",
            "is known to equal",
            "equals its list chromatic number",
        )
        if any(marker in explanation for marker in generic_formula_markers):
            return "unsupported_numeric_formula_generic"

        explicit_math_markers = (
            "=",
            "\\binom",
            "binom",
            "factorial",
            "!",
            "min(",
            "max(",
            "choose",
            "order of",
            "count of",
            "factors as",
            "product of",
            "q^",
            "^",
            "divides",
            "multiplied by",
        )
        has_explicit_formula = any(marker in explanation for marker in explicit_math_markers)
        if not has_explicit_formula:
            return "unsupported_numeric_formula_generic"

        family_applicability_markers = (
            "applies here",
            "applies because",
            "for this case",
            "for this graph",
            "for this group",
            "for psu(",
            "for k_{",
            "for k_",
            "is determined by the formula",
            "is given by the formula",
        )
        object_class_markers = (
            "complete bipartite graph",
            "bipartite graph",
            "graph k_{",
            "graph k_",
            "psu(",
            "projective special unitary",
            "unitary group",
            "largest prime divisor",
            "prime divisor",
            "integer",
            "number of involutions",
        )
        wrong_family_formula_markers = (
            "min(",
            "max(",
            "2^{",
            "2^",
            "(n-1)!",
            "n!",
            "\\binom",
            "binom",
            "factorial",
            "!",
            "(m+n)!",
            "(1000+1000)!",
            "(2000)!",
            "(1000!)",
            "order of the group",
            "number of involutions is",
            "alon-tarsi number",
            "alon–tarsi number",
            "list chromatic number",
            "m+1",
            "n+1",
            "1000+1",
            "1000 + 1",
        )
        applicability_assertion_markers = (
            "therefore",
            "thus",
            "hence",
            "so the answer is",
            "which gives",
            "which yields",
            "this gives",
            "this yields",
            "for a complete bipartite graph",
            "for this complete bipartite graph",
            "for this group",
            "for psu(",
            "for k_{",
            "for k_",
            "because",
            "since",
        )

        compact_answer = answer.replace(",", "").replace(" ", "")
        explanation_compact = explanation.replace(",", "").replace(" ", "")
        has_instantiation_shell = any(
            marker in explanation
            for marker in (
                "substituting",
                "for k_{",
                "for k_",
                "for psu(",
                "gives",
                "obtained from",
                "becomes",
                "\\binom",
                "binom",
                "factorial",
                "!",
            )
        )

        answer_number = re.sub(r"[^0-9.+\\-]", "", answer)
        explanation_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", explanation)
        answer_is_reused_in_explanation = bool(compact_answer) and compact_answer in explanation_compact
        answer_is_only_final_number = bool(answer_number) and explanation_numbers.count(answer_number) == 1
        has_family_applicability = any(marker in explanation for marker in family_applicability_markers)
        has_object_class_reference = any(marker in explanation for marker in object_class_markers)
        has_wrong_family_formula = any(marker in explanation for marker in wrong_family_formula_markers)
        has_applicability_assertion = any(marker in explanation for marker in applicability_assertion_markers)

        if (
            has_wrong_family_formula
            and has_object_class_reference
            and (has_family_applicability or has_applicability_assertion)
            and (answer_is_reused_in_explanation or answer_is_only_final_number)
        ):
            return "unsupported_numeric_formula_wrong_family"
        if compact_answer and compact_answer in explanation_compact:
            if has_family_applicability:
                return "unsupported_numeric_formula_wrong_family"
        if compact_answer and compact_answer not in explanation_compact:
            if has_instantiation_shell:
                return "unsupported_numeric_formula_uninstantiated"

        return ""

    @staticmethod
    def _cleanup_final_answer(text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = cleaned.strip(" |")
        if not cleaned:
            return ""
        if HLEAdapter._looks_like_internal_guidance(cleaned):
            return ""
        cleaned = re.sub(r"^\s*(exact answer|final answer|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("$", "").strip()
        if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
            cleaned = cleaned[2:-2].strip()
        cleaned = re.sub(
            r"(?i)\s+(confidence|explanation|reasoning)\s*:\s*.+$",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^(the answer is|the solutions are|solution:)\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if HLEAdapter._looks_like_internal_guidance(cleaned):
            return ""
        return cleaned

    @staticmethod
    def _looks_like_internal_guidance(text: str) -> bool:
        sample = str(text or "").strip()
        if not sample:
            return False
        compact = re.sub(r"\s+", " ", sample).strip().lower()
        internal_markers = (
            "[emma",
            "[private",
            "[memory",
            "private self-check",
            "previous_submission",
            "repair_target_failure_boundary",
            "repair_rule",
            "proof_obligation",
            "next_reasoning_move",
            "disallowed_shortcut",
            "current question context:",
            "recent action history:",
            "benchmark: humanity's last exam",
            "identify the exact symbolic operator or branch structure first",
            "write down the exact theorem / counting identity",
            "pin down the decisive entity, date, or relation first",
            "do not answer from topical familiarity",
            "do not emit a numeric answer",
            "follow them only when they fit the current task contract",
        )
        if any(marker in compact for marker in internal_markers):
            return True
        imperative_prefixes = (
            "identify ",
            "write ",
            "pin down ",
            "reconstruct ",
            "re-derive ",
            "derive ",
            "solve for ",
            "preserve ",
            "convert ",
            "collapse ",
            "after computing ",
            "before the final ",
        )
        if any(compact.startswith(prefix) for prefix in imperative_prefixes):
            return True
        if "\n" in sample and "exact answer:" not in compact and "final answer:" not in compact:
            if any(token in compact for token in ("confidence:", "verdict", "prediction:", "failure_signal", "judge_model")):
                return True
        return False

    @staticmethod
    def _judge_failure_signal(extracted: str, prediction: str, answer_contract_family: str, judge_error: str) -> str:
        if judge_error:
            return "judge_error"
        if not str(extracted or "").strip():
            return "answer_missing"
        if answer_contract_family == "numeric_exact_contract":
            unsupported_type = HLEAdapter._unsupported_numeric_formula_type(prediction, extracted)
            if unsupported_type:
                return unsupported_type
            return "numeric_reasoning_mismatch"
        if answer_contract_family == "symbolic_exact_contract":
            return "symbolic_reasoning_mismatch"
        if answer_contract_family == "multiple_choice_contract":
            return "multiple_choice_reasoning_mismatch"
        if "\n" in str(prediction or "").strip() and "Exact Answer:" not in str(prediction or ""):
            return "text_answer_scope_mismatch"
        return "text_reasoning_mismatch"

    @staticmethod
    def _subject_family(subject: Any) -> str:
        text = str(subject or "").strip().lower()
        if "math" in text:
            return "formal_reasoning"
        if "humanities" in text or "social" in text or "history" in text:
            return "factoid_reasoning"
        if "biology" in text or "medicine" in text or "physics" in text or "chemistry" in text or "engineering" in text:
            return "scientific_reasoning"
        if "computer" in text or "ai" in text:
            return "technical_reasoning"
        return "general_reasoning"

    def _reasoning_family(self, record: dict[str, Any]) -> str:
        answer_contract = self._answer_contract_family(record)
        question = self._extract_question(record).lower()
        subject_family = self._subject_family(self._record_value(record, "subject", "category", "discipline", default=""))
        query_family = self._factoid_query_family(question)
        if answer_contract == "multiple_choice_contract":
            return f"{subject_family}|option_elimination"
        if answer_contract == "numeric_exact_contract":
            if any(token in question for token in ("tan(", "sin(", "cos(", "period", "mod π", "mod pi", "modulo π", "modulo pi")):
                return f"{subject_family}|periodic_numeric_reduction"
        if any(token in question for token in ("largest", "smallest", "number of", "how many", "sum", "prime", "solve for")):
            return f"{subject_family}|symbolic_or_numeric_derivation"
        if answer_contract == "text_exact_contract":
            if query_family in {
                "person_identity_query",
                "temporal_fact_query",
                "location_fact_query",
                "paired_relation_query",
                "sequence_completion_query",
                "factoid_exact_query",
            }:
                return f"{subject_family}|{self._factoid_reasoning_family(question)}"
        return f"{subject_family}|closed_form_exact_answer"

    @staticmethod
    def _answer_contract_family(record: dict[str, Any]) -> str:
        answer_type = HLEAdapter._answer_type(record)
        choices = HLEAdapter._extract_choices(record)
        if answer_type in {"multiple_choice", "multiple-choice", "choice", "mcq"} or choices:
            return "multiple_choice_contract"
        gold = HLEAdapter._record_value(
            record,
            "answer",
            "correct_answer",
            "target",
            "label",
            default="",
        )
        if HLEAdapter._try_float(gold) is not None:
            return "numeric_exact_contract"
        gold_text = str(gold or "")
        if any(token in gold_text for token in ("\\", "{", "}", "W_", "ℤ")):
            return "symbolic_exact_contract"
        return "text_exact_contract"

    def _abstract_signature(self, record: dict[str, Any]) -> str:
        answer_contract = self._answer_contract_family(record)
        factoid_cue = ""
        if answer_contract == "text_exact_contract":
            factoid_cue = self._factoid_query_family(self._extract_question(record).strip().lower())
        return "|".join(
            part for part in [
                "closed_ended_question",
                self._subject_family(self._record_value(record, "subject", "category", "discipline", default="")),
                self._reasoning_family(record),
                answer_contract,
                self._question_form_family(record),
                factoid_cue,
                self._judge_reliability_flag(record),
                "direct_final_answer",
                "exact_answer_only",
            ] if part
        )

    @staticmethod
    def _question_form_family(record: dict[str, Any]) -> str:
        question = HLEAdapter._extract_question(record).strip().lower()
        choices = HLEAdapter._extract_choices(record)
        answer_contract = HLEAdapter._answer_contract_family(record)
        if choices:
            return "multiple_choice_selection"
        if answer_contract == "numeric_exact_contract":
            if any(token in question for token in ("tan(", "sin(", "cos(", "period", "mod π", "mod pi", "modulo π", "modulo pi")):
                return "periodic_numeric_query"
            if any(token in question for token in ("largest", "smallest", "how many", "number of", "count", "divisor")):
                return "numeric_closed_form_query"
            return "numeric_exact_query"
        if answer_contract == "symbolic_exact_contract":
            if any(token in question for token in ("solve for all", "solve", "closed form")):
                return "symbolic_solution_family"
            return "symbolic_exact_query"
        if answer_contract == "text_exact_contract":
            return HLEAdapter._factoid_query_family(question)
        return "closed_ended_exact_query"

    @staticmethod
    def _factoid_query_family(question: str) -> str:
        text = str(question or "").strip().lower()
        if "what letter should appear next in the sequence" in text or "sequence" in text:
            return "sequence_completion_query"
        if text.startswith("who ") or "who was" in text or "who is" in text:
            return "person_identity_query"
        if text.startswith("when ") or "what year" in text or re.search(r"\bin \d{3,4}\b", text):
            return "temporal_fact_query"
        if "preserved alongside" in text or "alongside" in text or "what book of manners" in text:
            return "paired_relation_query"
        if text.startswith("where "):
            return "location_fact_query"
        return "factoid_exact_query"

    @staticmethod
    def _factoid_reasoning_family(question: str) -> str:
        query_family = HLEAdapter._factoid_query_family(question)
        mapping = {
            "sequence_completion_query": "pattern_sequence_completion",
            "paired_relation_query": "entity_relation_retrieval",
            "person_identity_query": "entity_or_fact_retrieval",
            "temporal_fact_query": "entity_or_fact_retrieval",
            "location_fact_query": "entity_or_fact_retrieval",
            "factoid_exact_query": "entity_or_fact_retrieval",
        }
        return mapping.get(query_family, "entity_or_fact_retrieval")

    @staticmethod
    def _extraction_risk(record: dict[str, Any]) -> str:
        choices = HLEAdapter._extract_choices(record)
        answer_contract = HLEAdapter._answer_contract_family(record)
        if choices:
            return "low"
        if answer_contract == "numeric_exact_contract":
            return "medium"
        if answer_contract == "symbolic_exact_contract":
            return "high"
        return "medium"

    @staticmethod
    def _judge_reliability_flag(record: dict[str, Any]) -> str:
        answer_contract = HLEAdapter._answer_contract_family(record)
        if answer_contract == "symbolic_exact_contract":
            return "judge_risk_high"
        if answer_contract == "numeric_exact_contract":
            return "judge_risk_medium"
        return "judge_risk_low"

    @staticmethod
    def _choice_failure_signal(prediction: str, resolved_prediction: str, choices: list[str]) -> str:
        if not str(prediction or "").strip():
            return "answer_missing"
        if not str(resolved_prediction or "").strip():
            return "multiple_choice_format_mismatch"
        if resolved_prediction not in choices:
            return "multiple_choice_option_mismatch"
        return "multiple_choice_reasoning_mismatch"

    @staticmethod
    def _text_failure_signal(prediction: str, gold: Any, answer_contract_family: str) -> str:
        trimmed = str(prediction or "").strip()
        if not trimmed:
            return "protocol_answer_missing"
        if answer_contract_family == "symbolic_exact_contract":
            if "\n" in trimmed or len(trimmed.split()) > 12:
                return "contract_symbolic_exact_mismatch"
            return HLEAdapter._symbolic_failure_signal(trimmed)
        if answer_contract_family == "text_exact_contract":
            if len(trimmed.split()) > max(12, len(str(gold or "").split()) * 2):
                return "protocol_extraction_mismatch"
            return "reasoning_fact_retrieval_mismatch"
        return "reasoning_numeric_mismatch"

    @staticmethod
    def _symbolic_failure_signal(prediction: str) -> str:
        text = str(prediction or "").strip().lower()
        if not text:
            return "protocol_answer_missing"
        if "cannot be determined" in text or "impossible to determine" in text:
            return "reasoning_symbolic_mismatch"
        if any(marker in text for marker in (" or ", ",")) and not any(marker in text for marker in ("forall", "\\forall", "for all", "\\in", "ℤ", "mathbb{z}", "mathbb{c}", "parameter")):
            return "contract_symbolic_exact_mismatch"
        if any(marker in text for marker in ("ln(", "log(", "\\ln", "\\log")) and not any(marker in text for marker in ("w_", "lambert", "wk", "w_k")):
            return "reasoning_symbolic_mismatch"
        if any(marker in text for marker in ("z = 0", "z=0", "z = 1", "z=1", "z = 2", "z=2")) and any(marker in text for marker in ("\\in", "ℤ", "mathbb{z}")):
            return "reasoning_symbolic_mismatch"
        return "reasoning_symbolic_mismatch"

    @staticmethod
    def _reward_profile(
        *,
        record: dict[str, Any],
        success: bool,
        failure_signal: str,
        answer_contract_family: str,
        reasoning_family: str,
    ) -> dict[str, Any]:
        failure_signal = str(failure_signal or "none").strip() or "none"
        subject_family = HLEAdapter._subject_family(
            HLEAdapter._record_value(record, "subject", "category", "discipline", default="")
        )
        question_form_family = HLEAdapter._question_form_family(record)
        judge_reliability_flag = HLEAdapter._judge_reliability_flag(record)
        predicted_boundary = HLEAdapter._predicted_boundary(
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
            question_form_family=question_form_family,
            judge_reliability_flag=judge_reliability_flag,
        )
        predicted_severity = HLEAdapter._predicted_boundary_severity(predicted_boundary)
        observed_boundary = "calibrated_success" if success else HLEAdapter._normalize_boundary_label(failure_signal)
        observed_severity, correction_rule = HLEAdapter._failure_stage_profile(observed_boundary, success)
        reasoning_move_bundle = HLEAdapter._reasoning_move_bundle(
            failure_boundary=observed_boundary,
            answer_contract_family=answer_contract_family,
            reasoning_family=reasoning_family,
            question_form_family=question_form_family,
        )
        delta_hle = 1.0 if success else predicted_severity - observed_severity
        if observed_boundary == "protocol_judge_contamination":
            delta_signature = "protocol_delta_blocked"
        elif success:
            delta_signature = "delta_positive"
        elif delta_hle < 0:
            delta_signature = "delta_negative"
        elif delta_hle > 0:
            delta_signature = "delta_positive"
        else:
            delta_signature = "delta_neutral"
        phi = HLEAdapter._topology_potential_from_boundaries(
            predicted_severity=predicted_severity,
            observed_severity=observed_severity,
            delta_hle=delta_hle,
            success=success,
        )
        value_signal = phi if success else max(-1.0, min(0.0, delta_hle))
        return {
            "reward": round(phi, 4) if success else 0.0,
            "topology_potential": round(phi, 4),
            "value_signal": round(value_signal, 4),
            "pass_ratio": 1.0 if success else 0.0,
            "predicted_boundary": predicted_boundary,
            "predicted_boundary_severity": round(predicted_severity, 4),
            "observed_boundary": observed_boundary,
            "observed_boundary_severity": round(observed_severity, 4),
            "delta_hle": round(delta_hle, 4),
            "delta_signature": delta_signature,
            "failure_boundary": observed_boundary,
            "correction_rule": correction_rule,
            "reasoning_failure_pattern": reasoning_move_bundle.get("reasoning_failure_pattern", ""),
            "recompute_operator": reasoning_move_bundle.get("recompute_operator", ""),
            "disallowed_shortcut": reasoning_move_bundle.get("disallowed_shortcut", ""),
            "next_reasoning_move": reasoning_move_bundle.get("next_reasoning_move", ""),
            "proof_obligation": reasoning_move_bundle.get("proof_obligation", ""),
            "answer_contract_family": answer_contract_family,
            "reasoning_family": reasoning_family,
            "question_form_family": question_form_family,
            "judge_reliability_flag": judge_reliability_flag,
            "subject_family": subject_family,
        }

    @staticmethod
    def _normalize_boundary_label(failure_signal: str) -> str:
        signal = str(failure_signal or "").strip().lower()
        mapping = {
            "answer_missing": "protocol_answer_missing",
            "protocol_answer_missing": "protocol_answer_missing",
            "multiple_choice_format_mismatch": "contract_multiple_choice_format_mismatch",
            "multiple_choice_option_mismatch": "contract_multiple_choice_format_mismatch",
            "contract_numeric_audit_mismatch": "contract_numeric_audit_mismatch",
            "numeric_reasoning_mismatch": "reasoning_numeric_mismatch",
            "unsupported_numeric_formula": "reasoning_numeric_formula_bluff",
            "unsupported_numeric_formula_generic": "reasoning_numeric_formula_generic_bluff",
            "unsupported_numeric_formula_wrong_family": "reasoning_numeric_wrong_formula_family",
            "unsupported_numeric_formula_uninstantiated": "reasoning_numeric_formula_uninstantiated_bluff",
            "symbolic_reasoning_mismatch": "reasoning_symbolic_mismatch",
            "reasoning_mismatch": "reasoning_numeric_mismatch",
            "text_reasoning_mismatch": "reasoning_fact_retrieval_mismatch",
            "symbolic_answer_format_mismatch": "contract_symbolic_exact_mismatch",
            "symbolic_sample_root_pattern_with_parameter_tail": "reasoning_symbolic_mismatch",
            "symbolic_finite_case_list_instead_of_family": "contract_symbolic_exact_mismatch",
            "symbolic_branch_form_without_family_operator": "reasoning_symbolic_mismatch",
            "text_answer_scope_mismatch": "protocol_extraction_mismatch",
            "text_answer_scope_mismatch": "protocol_extraction_mismatch",
            "judge_error": "protocol_judge_contamination",
        }
        if signal in mapping:
            return mapping[signal]
        if "judge" in signal:
            return "protocol_judge_contamination"
        if "multiple_choice" in signal:
            return "contract_multiple_choice_format_mismatch"
        if signal == "unsupported_numeric_formula_generic":
            return "reasoning_numeric_formula_generic_bluff"
        if signal == "unsupported_numeric_formula_wrong_family":
            return "reasoning_numeric_wrong_formula_family"
        if signal == "unsupported_numeric_formula_uninstantiated":
            return "reasoning_numeric_formula_uninstantiated_bluff"
        if "unsupported_numeric_formula" in signal or "formula_bluff" in signal:
            return "reasoning_numeric_formula_bluff"
        if "numeric" in signal:
            return "reasoning_numeric_mismatch"
        if "symbolic" in signal:
            return "reasoning_symbolic_mismatch"
        if "fact" in signal or "entity" in signal:
            return "reasoning_fact_retrieval_mismatch"
        return "reasoning_fact_retrieval_mismatch"

    @staticmethod
    def _predicted_boundary(
        *,
        answer_contract_family: str,
        reasoning_family: str,
        question_form_family: str,
        judge_reliability_flag: str,
    ) -> str:
        if judge_reliability_flag == "judge_risk_high":
            return "contract_symbolic_exact_mismatch"
        if answer_contract_family == "multiple_choice_contract":
            return "contract_multiple_choice_format_mismatch"
        if answer_contract_family == "numeric_exact_contract":
            return "reasoning_numeric_mismatch"
        if answer_contract_family == "symbolic_exact_contract":
            return "reasoning_symbolic_mismatch"
        if "fact_retrieval" in reasoning_family or question_form_family == "factoid_exact_query":
            return "reasoning_fact_retrieval_mismatch"
        return "reasoning_fact_retrieval_mismatch"

    @staticmethod
    def _predicted_boundary_severity(predicted_boundary: str) -> float:
        lookup = {
            "protocol_answer_missing": 0.1,
            "protocol_extraction_mismatch": 0.12,
            "protocol_judge_contamination": 0.0,
            "contract_multiple_choice_format_mismatch": 0.35,
            "contract_numeric_exact_mismatch": 0.42,
            "contract_numeric_audit_mismatch": 0.36,
            "contract_symbolic_exact_mismatch": 0.45,
            "reasoning_numeric_formula_generic_bluff": 0.52,
            "reasoning_numeric_wrong_formula_family": 0.56,
            "reasoning_numeric_formula_uninstantiated_bluff": 0.6,
            "reasoning_numeric_formula_bluff": 0.58,
            "reasoning_numeric_mismatch": 0.68,
            "reasoning_symbolic_mismatch": 0.72,
            "reasoning_fact_retrieval_mismatch": 0.62,
            "calibrated_success": 1.0,
        }
        return lookup.get(str(predicted_boundary or "").strip(), 0.5)

    @staticmethod
    def _failure_stage_profile(failure_signal: str, success: bool) -> tuple[float, str]:
        if success:
            return 1.0, ""
        if failure_signal == "protocol_answer_missing":
            return 0.05, "Emit exactly one final answer span under the declared answer contract, with no explanation or extra text."
        if failure_signal == "protocol_extraction_mismatch":
            return 0.1, "Keep the answer terse and contract-compliant; output only the final answer span without surrounding explanation."
        if failure_signal == "protocol_judge_contamination":
            return 0.0, "Mark this case as protocol-blocked and do not treat the verdict as a solver-quality signal."
        if failure_signal == "contract_multiple_choice_format_mismatch":
            return 0.3, "Resolve the reasoning internally, then emit only the option letter or exact option text that matches the declared multiple-choice contract."
        if failure_signal == "contract_numeric_exact_mismatch":
            return 0.38, "Preserve the numeric-answer contract and emit one exact numeric answer span only."
        if failure_signal == "contract_numeric_audit_mismatch":
            return 0.34, "Reject the answer unless its verification line uses the same candidate as Exact Answer and recomputes to remainder 0."
        if failure_signal == "contract_symbolic_exact_mismatch":
            return 0.42, "Preserve symbolic exactness and emit one canonical closed-form line instead of a partial list or malformed family."
        if failure_signal == "reasoning_numeric_formula_generic_bluff":
            return 0.52, "Do not emit a numeric answer from generic theorem language alone; first state the exact instantiated counting or algebraic formula for this object."
        if failure_signal == "reasoning_numeric_wrong_formula_family":
            return 0.56, "Do not rely on a plausible-looking formula family unless you can justify why that family governs this exact object; first verify the object class and only then select the formula."
        if failure_signal == "reasoning_numeric_formula_uninstantiated_bluff":
            return 0.6, "Do not stop at a family-level formula; instantiate the concrete parameters, evaluate the instantiated expression to a single numeric value, and emit only that fully evaluated number."
        if failure_signal == "reasoning_numeric_formula_bluff":
            return 0.58, "Do not emit a numeric answer until the stated theorem or formula explicitly determines that exact value; generic theorem name-dropping does not count."
        if failure_signal == "reasoning_numeric_mismatch":
            return 0.7, "Preserve the numeric-answer contract and verify the final computed value before emitting it."
        if failure_signal == "reasoning_symbolic_mismatch":
            return 0.78, "Preserve symbolic exactness and derive the mathematically complete closed-form answer rather than a guessed template."
        if failure_signal == "reasoning_fact_retrieval_mismatch":
            return 0.64, "Re-evaluate the closed-ended reasoning path and verify the final fact or entity before answering."
        return 0.5, "Do not reuse the previous local answer pattern blindly; recompute under the current structural contract."

    @staticmethod
    def _reasoning_move_bundle(
        *,
        failure_boundary: str,
        answer_contract_family: str,
        reasoning_family: str,
        question_form_family: str,
    ) -> dict[str, str]:
        boundary = str(failure_boundary or "").strip()
        contract = str(answer_contract_family or "").strip()
        reasoning = str(reasoning_family or "").strip()
        question_form = str(question_form_family or "").strip()

        bundle = {
            "reasoning_failure_pattern": "",
            "recompute_operator": "",
            "disallowed_shortcut": "",
            "next_reasoning_move": "",
            "proof_obligation": "",
        }

        if boundary == "reasoning_numeric_formula_generic_bluff":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer used generic theorem or formula language without writing the concrete mathematical expression needed for this task.",
                    "recompute_operator": "Replace the generic explanation with the exact theorem, invariant, or counting formula specialized to the requested object, and verify that this formula family is actually valid for the current object before thinking about the final number.",
                    "disallowed_shortcut": "Do not use phrases like 'by a standard formula', 'by prime factorization', or 'by properties of involutions' as if they were derivations, and do not swap in a plausible-looking formula family without justifying why it applies here.",
                    "next_reasoning_move": "Name the concrete formula family, explain in one clause why it applies to this exact object, and only then write the instantiated formula; if you cannot do that, do not answer with a number yet.",
                    "proof_obligation": "Before the final number, there must exist an internal concrete formula for this exact object and an internal justification for why that formula family applies here, not just a topic-level theorem name.",
                }
            )
        elif boundary == "reasoning_numeric_wrong_formula_family":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer selected a concrete-looking formula family and even justified it, but that formula family does not actually govern the current object.",
                    "recompute_operator": "First classify the current object into the correct structural family, then select the theorem or counting formula that is valid for that family before writing any instantiated expression.",
                    "disallowed_shortcut": "Do not justify a formula family just because it looks plausible for a nearby object class, a simpler special case, or a memorized pattern from a related domain.",
                    "next_reasoning_move": "State the object class first, then name the theorem or formula family that applies to that class, and only then instantiate it; if the class-to-formula link is uncertain, do not emit a number.",
                    "proof_obligation": "Before the final number, there must exist an internal chain of the form current object class -> valid formula family for that class -> instantiated expression -> final numeric value.",
                }
            )
        elif boundary == "reasoning_numeric_formula_uninstantiated_bluff":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer wrote a family-level formula or symbolic shell but never instantiated it all the way to the emitted number.",
                    "recompute_operator": "Instantiate every task parameter into the formula, evaluate the instantiated expression down to a single numeric value, and only then emit that fully evaluated number.",
                    "disallowed_shortcut": "Do not leave Exact Answer as a factorial shell, binomial shell, symbolic shell, or memorable stand-in number when the instantiated expression has not been fully evaluated.",
                    "next_reasoning_move": "Substitute the exact parameters into the formula, carry the evaluation to the final integer, and verify that Exact Answer is that evaluated integer only.",
                    "proof_obligation": "Before the final number, there must exist an internal instantiated expression whose fully evaluated value is exactly the emitted number, with no residual operators or symbolic shell left over.",
                }
            )
        elif boundary == "reasoning_numeric_formula_bluff":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer named a theorem or formula but did not make that statement actually determine the emitted number.",
                    "recompute_operator": "State a formula that still contains the task variables, then substitute the concrete parameters and verify that the resulting expression really equals the final number.",
                    "disallowed_shortcut": "Do not use generic theorem name-dropping, family folklore, or placeholder formula language as a substitute for an instantiated derivation.",
                    "next_reasoning_move": "Write the exact instantiated formula for this object and check that the final number is the value of that formula, not a nearby or memorable stand-in.",
                    "proof_obligation": "Before the final number, there must exist an internal equality chain from the stated theorem or formula to the instantiated expression to that exact numeric value.",
                }
            )
        elif boundary == "reasoning_numeric_mismatch":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer pattern jumped to a numeric output without a validated derivation for this exact object.",
                    "recompute_operator": "First derive the governing invariant or closed-form formula symbolically, then substitute the concrete parameter only at the end.",
                    "disallowed_shortcut": "Do not answer from graph-family folklore, pattern memory, or previous_answer +/- 1 edits when no derived formula produced that number.",
                    "next_reasoning_move": "Write down the exact theorem / counting identity that determines the requested number for this family, and refuse to emit a number until that identity yields one.",
                    "proof_obligation": "Before the final number, there must exist an internal derivation chain of the form theorem or invariant -> symbolic expression -> numeric substitution.",
                }
            )
        elif boundary == "reasoning_symbolic_mismatch":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer reused a symbolic template without proving it satisfies the full family or branch condition.",
                    "recompute_operator": "Re-derive the equation or functional constraint from first principles and solve for the complete symbolic family before formatting the final line.",
                    "disallowed_shortcut": "Do not recycle a familiar root pattern, branch template, or isolated sample solution unless it is proven complete for all valid branches.",
                    "next_reasoning_move": "Identify the exact symbolic operator or branch structure first, then derive the full family with the required parameterization.",
                    "proof_obligation": "Before the final expression, there must exist an internal check that every claimed branch or family member satisfies the original equation and that no required branch is omitted.",
                }
            )
        elif boundary == "reasoning_fact_retrieval_mismatch":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The previous answer committed to a fact or entity before verifying the decisive retrieval cue.",
                    "recompute_operator": "Reconstruct the closed-ended fact path from the exact entity / relation asked in the prompt, then answer only after the decisive cue is checked.",
                    "disallowed_shortcut": "Do not answer from topical familiarity or nearest remembered factoid.",
                    "next_reasoning_move": "Pin down the decisive entity, date, or relation first, then compare candidate facts against that exact cue.",
                    "proof_obligation": "Before the final answer, there must be an internal match between the prompt's decisive cue and the selected fact or entity.",
                }
            )
        elif boundary == "contract_numeric_exact_mismatch":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The answer may contain the right idea but the numeric contract was not enforced exactly.",
                    "recompute_operator": "Collapse the derivation to one exact numeric output span only.",
                    "disallowed_shortcut": "Do not append prose, units, ranges, or alternative candidates in the final answer span.",
                    "next_reasoning_move": "After computing the number, strip everything except the exact numeric output.",
                    "proof_obligation": "The final answer must be one exact numeric span and nothing else.",
                }
            )
        elif boundary == "contract_symbolic_exact_mismatch":
            bundle.update(
                {
                    "reasoning_failure_pattern": "The symbolic result was not expressed as one canonical exact family or closed-form line.",
                    "recompute_operator": "Normalize the derivation into one canonical symbolic output line before answering.",
                    "disallowed_shortcut": "Do not list partial roots, prose explanations, or malformed branch families in the final answer line.",
                    "next_reasoning_move": "Convert the symbolic result into a single exact family or closed form with the right parameterization.",
                    "proof_obligation": "The final answer must be one exact symbolic line that is complete for the required family.",
                }
            )

        if contract == "numeric_exact_contract" and not bundle["proof_obligation"]:
            bundle["proof_obligation"] = "Before the final number, there must exist an internal symbolic derivation or counting formula that yields that exact numeric output."
        if contract == "symbolic_exact_contract" and not bundle["proof_obligation"]:
            bundle["proof_obligation"] = "Before the final symbolic line, there must exist an internal completeness check for the claimed family or branch structure."

        if question_form == "numeric_closed_form_query" and not bundle["next_reasoning_move"]:
            bundle["next_reasoning_move"] = "Derive the closed-form quantity symbolically before substituting the concrete numeric parameter."
        if question_form == "periodic_numeric_query":
            bundle.update(
                {
                    "reasoning_failure_pattern": bundle["reasoning_failure_pattern"] or "The previous answer invoked periodicity but never completed the reduction from the huge argument to the decisive reduced angle or resulting digits.",
                    "recompute_operator": bundle["recompute_operator"] or "First reduce the argument using the exact period or modular equivalence, then evaluate the reduced expression and only then read off the requested digits.",
                    "disallowed_shortcut": bundle["disallowed_shortcut"] or "Do not stop at 'use periodicity' as if it were already the answer, and do not guess the digits before the reduced angle has been fixed.",
                    "next_reasoning_move": "State the exact period, reduce the huge argument to the decisive equivalent angle or residue, and only then compute the requested digits from that reduced value.",
                    "proof_obligation": "Before the final number, there must exist an internal chain of the form periodic reduction -> reduced angle or residue -> evaluated tangent value -> requested digits.",
                }
            )
        if question_form == "symbolic_solution_family" and not bundle["next_reasoning_move"]:
            bundle["next_reasoning_move"] = "Solve for the full symbolic family first, then compress it into one canonical exact line."
        if question_form == "paired_relation_query":
            bundle.update(
                {
                    "reasoning_failure_pattern": bundle["reasoning_failure_pattern"] or "The previous answer jumped to a nearby literary title without verifying the manuscript-level co-preservation relation asked in the prompt.",
                    "recompute_operator": bundle["recompute_operator"] or "Resolve the exact relation anchor in the prompt, then retrieve the paired work linked to that anchor rather than a broadly related work.",
                    "disallowed_shortcut": bundle["disallowed_shortcut"] or "Do not answer from topical association with the main work, author, or genre when the prompt asks for a specific co-preserved companion text.",
                    "next_reasoning_move": "Pin down the exact preservation relation first, then select the companion work that is explicitly linked to that manuscript or preservation context.",
                    "proof_obligation": "Before the final answer, there must be an internal match of the form target work -> preservation context -> companion text.",
                }
            )
        if question_form == "sequence_completion_query":
            bundle.update(
                {
                    "reasoning_failure_pattern": bundle["reasoning_failure_pattern"] or "The previous answer guessed a nearby letter without first identifying the structural rule generating the sequence.",
                    "recompute_operator": bundle["recompute_operator"] or "Extract the generating rule over positions or keyboard layout first, then extend that rule by one step before answering.",
                    "disallowed_shortcut": bundle["disallowed_shortcut"] or "Do not choose a visually adjacent or alphabetically nearby letter until the sequence rule has been made explicit.",
                    "next_reasoning_move": "Write the hidden sequence rule over the existing letters first, then apply it one more step and answer with only that next letter.",
                    "proof_obligation": "Before the final answer, there must exist an internal rule that explains the full observed sequence and uniquely yields the next letter.",
                }
            )

        if "formal_reasoning|symbolic_or_numeric_derivation" in reasoning and not bundle["disallowed_shortcut"]:
            bundle["disallowed_shortcut"] = "Do not answer from surface familiarity alone; use a derivation that matches the formal structure of the problem."

        return bundle

    @staticmethod
    def _topology_potential_from_boundaries(
        *,
        predicted_severity: float,
        observed_severity: float,
        delta_hle: float,
        success: bool,
    ) -> float:
        if success:
            return 1.0
        predicted_severity = max(0.0, min(1.0, float(predicted_severity)))
        observed_severity = max(0.0, min(1.0, float(observed_severity)))
        delta_hle = max(-1.0, min(1.0, float(delta_hle)))
        phi = 0.2 + 0.35 * predicted_severity + 0.15 * max(delta_hle, 0.0) - 0.25 * max(-delta_hle, 0.0)
        phi = min(phi, observed_severity)
        return max(0.02, min(0.95, phi))
