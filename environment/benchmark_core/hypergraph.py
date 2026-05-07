from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from .adapter import TaskSpec


def _normalize_action(action: str) -> str:
    text = (action or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text.startswith("action:"):
        text = text.split(":", 1)[1].strip()
    return text


def _token_overlap(lhs: str, rhs: str) -> float:
    lhs_tokens = set(re.findall(r"[a-z0-9_]+", (lhs or "").lower()))
    rhs_tokens = set(re.findall(r"[a-z0-9_]+", (rhs or "").lower()))
    if not lhs_tokens or not rhs_tokens:
        return 0.0
    return len(lhs_tokens & rhs_tokens) / max(1, min(len(lhs_tokens), len(rhs_tokens)))


def _history_actions(history_lines: list[str]) -> list[str]:
    actions: list[str] = []
    for item in history_lines:
        if item.startswith("> "):
            actions.append(_normalize_action(item[2:]))
    return actions


class HypergraphPromptAdapter:
    """
    Minimal structure layer for the shared benchmark backend.

    `full` uses memory records as navigable action chains with positive next-step
    recommendations and negative cautionary reminders. `no_HyperGraph` removes
    this layer and falls back to flat summary retrieval only.
    """

    def recommend(
        self,
        memory_bank: Any,
        task: TaskSpec,
        history_lines: list[str],
        valid_actions: list[str] | None,
        limit: int = 4,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        if memory_bank is None or not getattr(memory_bank, "records", None):
            return {"positive": [], "cautionary": []}

        if task.task_type == "code_generation":
            return self._recommend_code_graph(
                memory_bank=memory_bank,
                task=task,
                limit=limit,
                retrieval_context=retrieval_context,
            )

        past_actions = _history_actions(history_lines)
        positive: dict[str, tuple[float, str]] = {}
        cautionary: dict[str, tuple[float, str]] = {}

        for record in memory_bank.records:
            meta = record.get("meta") or {}
            action_sequence = [_normalize_action(item) for item in meta.get("action_sequence", []) if item]
            if not action_sequence:
                continue

            task_score = self._task_match_score(task, meta)
            if task_score <= 0.0:
                continue

            cursor = self._cursor(past_actions, action_sequence)
            if cursor >= len(action_sequence):
                continue

            next_action = action_sequence[cursor]
            q_value = float(record.get("q", 0.0))
            success = bool(meta.get("success", False))
            support = task_score * (0.7 + math.tanh(abs(q_value) / 6.0))

            if valid_actions:
                matched = self._best_valid_action(next_action, valid_actions)
                if matched is None:
                    continue
                next_action = matched

            reason = self._reason_text(meta, cursor, q_value)
            if success and q_value >= 0:
                if support > positive.get(next_action, (-1e9, ""))[0]:
                    positive[next_action] = (support, reason)
            elif not success and q_value <= 0:
                if support > cautionary.get(next_action, (-1e9, ""))[0]:
                    cautionary[next_action] = (support, reason)

        return {
            "positive": [f"{action} | {reason}" for action, (_, reason) in sorted(positive.items(), key=lambda item: item[1][0], reverse=True)[:limit]],
            "cautionary": [f"{action} | {reason}" for action, (_, reason) in sorted(cautionary.items(), key=lambda item: item[1][0], reverse=True)[:limit]],
        }

    def _recommend_code_graph(
        self,
        memory_bank: Any,
        task: TaskSpec,
        limit: int = 4,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        target_signature = str(task.metadata.get("abstract_signature", "")).strip()
        target_transform = str(task.metadata.get("transform_family", "")).strip()
        target_output = str(task.metadata.get("output_contract_family", "")).strip()
        retrieval_mode = str((retrieval_context or {}).get("retrieval_mode", "")).strip()
        used_memory_idx = (retrieval_context or {}).get("used_memory_idx")

        positive: dict[str, float] = {}
        cautionary: dict[str, float] = {}
        candidate_records: list[tuple[int, Any]] = []
        if retrieval_mode in {"core_match", "seed_core_match"} and isinstance(used_memory_idx, int):
            if 0 <= used_memory_idx < len(memory_bank.records):
                candidate_records.append((used_memory_idx, memory_bank.records[used_memory_idx]))
        elif retrieval_mode == "reference_only":
            # Reference-only memories can expose failure boundaries, but they
            # should not reopen positive structural guidance during local repair.
            if isinstance(used_memory_idx, int) and 0 <= used_memory_idx < len(memory_bank.records):
                candidate_records.append((used_memory_idx, memory_bank.records[used_memory_idx]))
        else:
            return {"positive": [], "cautionary": []}

        for record_idx, record in candidate_records:
            structured = record.get("s") or {}
            meta = record.get("meta") or {}
            if not isinstance(structured, dict):
                continue
            if str(structured.get("task_type", "")).strip() != task.task_type:
                continue
            if bool(meta.get("pruned", False)):
                continue

            match = 0.0
            memory_level = str(structured.get("memory_level", "")).strip()
            if memory_level == "strategy":
                if target_transform and str(structured.get("action_type", "")).strip() == target_transform:
                    match += 0.45
                if target_output and str(structured.get("output_contract_family", "")).strip() == target_output:
                    match += 0.25
                if str(structured.get("visualization_family", "")).strip() == str(task.metadata.get("visualization_family", "")).strip():
                    match += 0.15
                if str(structured.get("input_family", "")).strip() == str(task.metadata.get("input_family", "")).strip():
                    match += 0.15
            else:
                if target_signature and str(structured.get("abstract_signature", "")).strip() == target_signature:
                    match += 0.6
                if target_transform and str(structured.get("action_type", "")).strip() == target_transform:
                    match += 0.25
                if target_output and str(structured.get("output_contract_family", "")).strip() == target_output:
                    match += 0.15
            if match <= 0.0:
                continue

            q_value = float(record.get("q", 0.0))
            support = match * (0.7 + math.tanh(abs(q_value) / 6.0))
            positive_compatible = self._is_positive_code_match(task=task, structured=structured, memory_level=memory_level)
            if (
                retrieval_mode in {"core_match", "seed_core_match"}
                and structured.get("outcome") == "success"
                and q_value >= 0
                and str(structured.get("value_bias", "")).strip() == "positive_reuse"
                and bool(structured.get("trusted", memory_level != "strategy"))
                and positive_compatible
            ):
                key = (
                    f"{memory_level or 'tactical'}: {structured.get('input_family', '')} -> {structured.get('action_type', '')} "
                    f"-> {structured.get('output_contract_family', '')} with {structured.get('visualization_family', '')}"
                )
                success_contract_rule = str(structured.get("success_contract_rule", "")).strip()
                if success_contract_rule:
                    key = f"{key} | success contract: {success_contract_rule}"
                positive[key] = max(positive.get(key, -1e9), support)
            elif (
                structured.get("outcome") == "failure"
                and q_value <= 0
                and self._is_failure_code_match(task=task, structured=structured)
            ):
                failure_boundary = str(structured.get("failure_boundary", "")).strip() or "unknown_failure"
                correction_rule = str(structured.get("correction_rule", "")).strip()
                key = f"avoid failure boundary: {failure_boundary}"
                if correction_rule:
                    key = f"{key} | repair rule: {correction_rule}"
                cautionary[key] = max(cautionary.get(key, -1e9), support)

        return {
            "positive": [item for item, _ in sorted(positive.items(), key=lambda item: item[1], reverse=True)[:limit]],
            "cautionary": [item for item, _ in sorted(cautionary.items(), key=lambda item: item[1], reverse=True)[:limit]],
        }

    @staticmethod
    def _edge_support_score(structured: dict[str, Any], q_value: float, match_score: float) -> float:
        evidence = structured.get("evidence", {}) if isinstance(structured.get("evidence"), dict) else {}
        phi = float(evidence.get("topology_potential", 0.0) or 0.0)
        phi = max(0.0, min(1.0, phi))
        q_strength = min(1.0, abs(float(q_value)))
        memory_level = str(structured.get("memory_level", "")).strip()
        trusted = bool(structured.get("trusted", False))
        confidence = 0.55 + 0.25 * q_strength + 0.2 * phi
        level_bonus = 1.0
        if memory_level == "strategy":
            level_bonus += 0.05
        if trusted:
            level_bonus += 0.05
        return float(match_score) * confidence * level_bonus

    @staticmethod
    def _is_positive_code_match(task: TaskSpec, structured: dict[str, Any], memory_level: str) -> bool:
        task_meta = getattr(task, "metadata", {}) or {}

        def exact_match(task_key: str, memory_key: str | None = None) -> bool:
            memory_key = memory_key or task_key
            lhs = str(task_meta.get(task_key, "")).strip()
            rhs = str(structured.get(memory_key, "")).strip()
            return bool(lhs and rhs and lhs == rhs)

        # Positive reuse is only safe when the abstract decision structure is
        # genuinely aligned. Otherwise, even seemingly-related code tasks can
        # leak the wrong renderer or transform family into the prompt.
        required_matches = [
            exact_match("input_family"),
            exact_match("output_contract_family"),
            exact_match("visualization_family"),
            exact_match("visualization_contract_family"),
            exact_match("render_pattern_family"),
            exact_match("exception_contract_family"),
            exact_match("transform_family", "action_type"),
        ]

        for strict_key in ("plot_cardinality_rule", "axes_semantic_rule", "return_container_rule"):
            task_value = str(task_meta.get(strict_key, "")).strip()
            memory_value = str(structured.get(strict_key, "")).strip()
            if task_value or memory_value:
                required_matches.append(bool(task_value and memory_value and task_value == memory_value))

        if not all(required_matches):
            return False

        if memory_level == "tactical":
            return exact_match("abstract_signature")

        return True

    @staticmethod
    def _is_failure_code_match(task: TaskSpec, structured: dict[str, Any]) -> bool:
        task_meta = getattr(task, "metadata", {}) or {}

        def exact_match(task_key: str, memory_key: str | None = None) -> bool:
            memory_key = memory_key or task_key
            lhs = str(task_meta.get(task_key, "")).strip()
            rhs = str(structured.get(memory_key, "")).strip()
            return bool(lhs and rhs and lhs == rhs)

        required_matches = [
            exact_match("output_contract_family"),
            exact_match("visualization_family"),
            exact_match("visualization_contract_family"),
            exact_match("render_pattern_family"),
            exact_match("transform_family", "action_type"),
        ]
        for strict_key in ("plot_cardinality_rule", "axes_semantic_rule", "return_container_rule"):
            task_value = str(task_meta.get(strict_key, "")).strip()
            memory_value = str(structured.get(strict_key, "")).strip()
            if task_value or memory_value:
                required_matches.append(bool(task_value and memory_value and task_value == memory_value))
        return all(required_matches)

    def build_prompt_block(self, recommendations: dict[str, list[str]]) -> str:
        positive = recommendations.get("positive", [])
        cautionary = recommendations.get("cautionary", [])
        if not positive and not cautionary:
            return ""

        parts: list[str] = []
        if positive:
            parts.append("EMMA Structure Recommendations:")
            for idx, item in enumerate(positive, 1):
                parts.append(f"{idx}. {item}")
            parts.append("Treat every success contract above as an abstract structural check before committing to the final answer.")
        if cautionary:
            parts.append("EMMA Failure Boundary Warnings:")
            for idx, item in enumerate(cautionary, 1):
                parts.append(f"{idx}. Avoid repeating: {item}")
        parts.append("Use these only as abstract decision-structure hints. Follow them only when they fit the current task contract.")
        return "\n".join(parts)

    @staticmethod
    def _task_match_score(task: TaskSpec, meta: dict[str, Any]) -> float:
        meta_task_type = meta.get("task_type", "")
        meta_task_description = meta.get("task_description", "") or meta.get("instruction", "")

        if task.task_type and meta_task_type and task.task_type != meta_task_type:
            return 0.0

        overlap = _token_overlap(task.task_description or task.instruction, meta_task_description)
        if overlap <= 0.0:
            return 0.3 if meta_task_type == task.task_type else 0.0
        return min(1.0, 0.45 + overlap)

    @staticmethod
    def _cursor(history_actions: list[str], action_sequence: list[str]) -> int:
        if not history_actions:
            return 0

        max_len = min(len(history_actions), len(action_sequence))
        for length in range(max_len, 0, -1):
            if history_actions[-length:] == action_sequence[:length]:
                return length

        last_action = history_actions[-1]
        for idx, action in enumerate(action_sequence):
            if action == last_action:
                return idx + 1
        return 0

    @staticmethod
    def _best_valid_action(target_action: str, valid_actions: list[str]) -> str | None:
        best_action = None
        best_score = 0.0
        target_norm = _normalize_action(target_action)
        for action in valid_actions:
            score = _token_overlap(target_norm, _normalize_action(action))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_score >= 0.72 else None

    @staticmethod
    def _reason_text(meta: dict[str, Any], cursor: int, q_value: float) -> str:
        task_label = meta.get("task_type") or "same-task"
        total_steps = len(meta.get("action_sequence", []))
        return f"{task_label} memory q={q_value:.2f}, step {cursor + 1}/{max(total_steps, 1)}"
