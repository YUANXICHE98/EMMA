from __future__ import annotations

import gzip
import importlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class BigCodeBenchAdapter(BenchmarkAdapter):
    benchmark_name = "bigcodebench"
    _DEFAULT_DATASET_URLS = {
        "full": "https://raw.githubusercontent.com/bigcode-project/bigcodebench-annotation/main/hf_data/BigCodeBench.jsonl.gz",
    }

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.runner_cfg = config.get("runner", {})
        self.task_records: list[dict[str, Any]] = []
        self.current_record: dict[str, Any] | None = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self._gradio_client = None

    def setup(self) -> None:
        dataset_path = self._ensure_dataset()
        self.task_records = self._load_records(dataset_path)
        self.task_records = self._apply_selection(self.task_records)
        if not self.task_records:
            raise RuntimeError(f"BigCodeBench adapter loaded zero tasks from {dataset_path}")

    def reset_task(self, index: int | None = None) -> ResetResult:
        if not self.task_records:
            raise RuntimeError("BigCodeBench adapter has no loaded tasks.")

        task_index = 0 if index is None else index
        if task_index < 0 or task_index >= len(self.task_records):
            raise IndexError(f"BigCodeBench task index out of range: {task_index} / {len(self.task_records)}")

        record = self.task_records[task_index]
        self.current_record = record
        task_profile = self._task_profile(record)
        goal_repr = self._goal_repr(record, task_profile)
        self.current_task = TaskSpec(
            task_id=record["task_id"],
            instruction=record["instruct_prompt"],
            task_type="code_generation",
            task_description=record["instruct_prompt"],
            goal_repr=goal_repr,
            metadata={
                "benchmark": self.benchmark_name,
                "task_id": record["task_id"],
                "entry_point": record["entry_point"],
                "subset": self._subset(),
                "split": self._split(),
                "task_index": task_index,
                **task_profile,
            },
        )
        self.current_observation = self._render_observation(record)
        return ResetResult(
            task=self.current_task,
            observation=self.current_observation,
            state_repr=self._state_repr(record, task_profile),
            candidate_actions=self.get_valid_actions(),
            valid_actions=self.get_valid_actions(),
        )

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-4:]) if history_lines else "None"
        metadata = task.metadata
        entry_point = metadata.get("entry_point", "")
        abstract_goal = metadata.get("abstract_goal", "")
        projection_contracts = metadata.get("projection_contracts", "")
        contract_constraints = metadata.get("contract_constraints", "")
        render_pattern_hint = metadata.get("render_pattern_hint", "")
        evaluator_contract_note = metadata.get("evaluator_contract_note", "")
        lexical_bans = metadata.get("lexical_bans", "")
        visualization_contract_family = metadata.get("visualization_contract_family", "")
        task_context = self._render_observation(self.current_record) if isinstance(self.current_record, dict) else observation
        verifier_feedback = ""
        if observation and observation != task_context:
            verifier_feedback = f"\n\nCurrent verifier feedback:\n{observation}"
        exact_label_check = ""
        if visualization_contract_family == "explicit_plot_labels_and_titles":
            exact_label_check = (
                "\n[Exact Plot Label Check]\n"
                "This task uses a literal plot-text contract.\n"
                "Before submitting code, extract every required plot title, xlabel, and ylabel string from the task instruction and verify that the code sets those exact strings literally.\n"
                "Do not paraphrase them, do not shorten them, and do not replace them with synonyms.\n"
                "If the instruction requires a label like 'Frequency', the final code must use 'Frequency' exactly.\n"
            )
        rule_fields = [
            ("column_preservation_rule", metadata.get("column_preservation_rule", "")),
            ("matrix_operation_rule", metadata.get("matrix_operation_rule", "")),
            ("aggregation_rule", metadata.get("aggregation_rule", "")),
            ("plot_cardinality_rule", metadata.get("plot_cardinality_rule", "")),
            ("axes_semantic_rule", metadata.get("axes_semantic_rule", "")),
            ("return_container_rule", metadata.get("return_container_rule", "")),
            ("return_slot_signature", metadata.get("return_slot_signature", "")),
            ("plot_title_rule", metadata.get("plot_title_rule", "")),
            ("forbidden_patterns", metadata.get("forbidden_patterns", "")),
        ]
        compact_rules = "\n".join(
            f"[{name}]\n{value}"
            for name, value in rule_fields
            if str(value).strip()
        )
        return (
            "Benchmark: BigCodeBench\n"
            f"Split: {self._split()}\n"
            f"Subset: {self._subset()}\n"
            f"Task ID: {task.task_id}\n"
            f"Entry point: {entry_point}\n\n"
            "Write Python code for the benchmark task.\n"
            "Output only raw Python code, with no markdown fences and no explanation.\n"
            "Use the task instruction and starter code exactly as given.\n"
            "Return the full self-contained snippet, including the import lines and the function signature from the starter code.\n"
            "Never return only an indented function body or a partial continuation.\n"
            "Do not output analysis prose.\n\n"
            "Abstract interface:\n"
            f"[abstract_goal]\n{abstract_goal}\n\n"
            f"[projection_contracts]\n{projection_contracts}\n\n"
            f"{compact_rules}\n\n"
            f"[render_pattern_hint]\n{render_pattern_hint}\n\n"
            f"[evaluator_contract_note]\n{evaluator_contract_note}\n\n"
            f"[contract_constraints]\n{contract_constraints}\n\n"
            f"Task context:\n{task_context}"
            f"{verifier_feedback}\n\n"
            f"Recent action history:\n{history_block}\n\n"
            f"[Lexical Guardrails]\n{lexical_bans}\n\n"
            "[Submission Check]\n"
            "Before returning code, verify that the final code obeys every output contract, return-container rule, and lexical guardrail above.\n"
            "Check the returned slot types literally: if slot2 requires a single Axes, do not return a tuple/grid/container; if slot2 requires a histogram tuple or axes collection, do not collapse it into one Axes.\n"
            "Check output thickness literally: if the task requires a one-column summary dataframe, do not copy the original input table and append one extra column.\n"
            "If any banned pattern still appears in the code, rewrite it before submission.\n"
            "Treat lexical bans as hard invalidation rules, not as soft preferences."
            f"{exact_label_check}"
        )

    def step(self, action: str) -> StepResult:
        if self.current_record is None or self.current_task is None:
            raise RuntimeError("BigCodeBench step called before reset_task.")

        lexical_violation = self._detect_lexical_ban_violation(action, self.current_task)
        if lexical_violation is not None:
            observation = self._render_eval_observation(
                "fail",
                {"pass@1": 0.0},
                lexical_violation["reward_profile"],
            )
            self.current_observation = observation
            return StepResult(
                observation=observation,
                reward=lexical_violation["reward_profile"]["reward"],
                done=True,
                success=False,
                state_repr=self._eval_state_repr(status="fail", pass_at_k={"pass@1": 0.0}),
                candidate_actions=self.get_valid_actions(),
                failure_signal="lexical_ban_violation",
                terminal_status="failure",
                info=lexical_violation,
                valid_actions=self.get_valid_actions(),
            )

        try:
            results, pass_at_k = self._evaluate_solution(self.current_record, action)
            status = self._extract_status(results, self.current_record["task_id"])
            success = status == "pass"
            if status == "missing":
                reward_profile = self._build_recovered_reward_profile(
                    task=self.current_task,
                    action=action,
                    error_text="missing evaluation result",
                    pass_ratio=0.0,
                )
            else:
                reward_profile = self._reward_profile(results, self.current_record["task_id"], success)
            reward_profile["correction_rule"] = self._task_aware_correction_rule(
                reward_profile.get("failure_boundary", ""),
                str(reward_profile.get("correction_rule", "")),
            )
            raw_feedback = self._raw_feedback_summary(results, self.current_record["task_id"])
            if raw_feedback:
                reward_profile["raw_feedback_summary"] = raw_feedback
            reward = reward_profile["reward"]
            info = {
                "status": status,
                "pass_at_k": pass_at_k,
                "raw_results": results,
                "raw_feedback_summary": raw_feedback,
                "reward_profile": reward_profile,
            }
            observation = self._render_eval_observation(status, pass_at_k, reward_profile)
        except Exception as exc:
            raw_feedback = str(exc)
            success = False
            reward = 0.0
            reward_profile = self._build_recovered_reward_profile(
                task=self.current_task,
                action=action,
                error_text=raw_feedback,
                pass_ratio=0.0,
            )
            reward_profile["raw_feedback_summary"] = raw_feedback[:2000]
            info = {
                "status": "error",
                "error": raw_feedback,
                "raw_feedback_summary": raw_feedback[:2000],
                "reward_profile": reward_profile,
            }
            observation = f"BigCodeBench evaluator error: {exc}"

        self.current_observation = observation
        return StepResult(
            observation=observation,
            reward=reward,
            done=True,
            success=success,
            state_repr=self._eval_state_repr(status=info.get("status", "error"), pass_at_k=info.get("pass_at_k")),
            candidate_actions=self.get_valid_actions(),
            failure_signal=self._failure_signal(info),
            terminal_status="success" if success else "failure",
            info=info,
            valid_actions=self.get_valid_actions(),
        )

    def force_finish(self) -> StepResult:
        return StepResult(
            observation="BigCodeBench episode terminated before a code solution was submitted.",
            reward=0.0,
            done=True,
            success=False,
            state_repr="[code_state]\nsubmission_missing",
            candidate_actions=self.get_valid_actions(),
            failure_signal="submission_missing",
            terminal_status="forced_terminate",
            info={"forced_terminate": True},
            valid_actions=self.get_valid_actions(),
        )

    def normalize_action(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if not action:
            return ""
        if action.startswith("```"):
            lines = [line for line in action.splitlines() if not line.strip().startswith("```")]
            action = "\n".join(lines).strip()
        lines = action.splitlines()
        cleaned: list[str] = []
        seen_code = False
        for line in lines:
            stripped = line.strip()
            if not seen_code:
                if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
                    seen_code = True
                    cleaned.append(line)
                continue
            if stripped.startswith(("But ", "Wait,", "However,", "So ", "The code", "This code")):
                break
            cleaned.append(line)
        if cleaned:
            action = "\n".join(cleaned).strip()
        return action

    @staticmethod
    def _detect_lexical_ban_violation(action: str, task: TaskSpec) -> dict[str, Any] | None:
        metadata = getattr(task, "metadata", {}) or {}
        lexical_bans = str(metadata.get("lexical_bans", "")).strip()
        if not lexical_bans:
            return None

        banned_tokens = []
        if ":" in lexical_bans:
            banned_tokens = [token.strip() for token in lexical_bans.split(":", 1)[1].split(";") if token.strip()]
        else:
            banned_tokens = [token.strip() for token in lexical_bans.split(";") if token.strip()]

        matched = [token for token in banned_tokens if token and token in action]
        if not matched:
            return None

        correction_rule = (
            "Lexical hard-ban violation: remove the forbidden pattern(s) exactly as listed before submission. "
            "Rewrite the code so none of these banned substrings remain: " + ", ".join(matched) + "."
        )
        return {
            "status": "fail",
            "pass_at_k": {"pass@1": 0.0},
            "raw_results": {
                "local_validation": {
                    "matched_lexical_bans": matched,
                }
            },
            "reward_profile": {
                "reward": 0.12,
                "topology_potential": 0.12,
                "value_signal": -0.88,
                "pass_ratio": 0.0,
                "passed_tests": 0,
                "failed_tests": 1,
                "total_tests": 1,
                "failure_boundary": "lexical_ban_violation",
                "correction_rule": correction_rule,
            },
        }

    def get_valid_actions(self) -> list[str]:
        return ["submit_python_solution"]

    def seed_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "state_text": (
                    "[task_type=code_generation]\n"
                    "[goal]\n"
                    "Map dataframe_input into columnwise_standardization with histogram output and return a "
                    "dataframe plus axes collection under a dataframe-native histogram renderer.\n\n"
                    "[task_domain_family]\n"
                    "dataframe_visualization\n\n"
                    "[input_family]\n"
                    "dataframe_input\n\n"
                    "[column_preservation_rule]\n"
                    "preserve the original dataframe column set during numeric preprocessing; do not drop all-NaN "
                    "intended numeric columns via select_dtypes filtering\n\n"
                    "[transform_family]\n"
                    "columnwise_standardization\n\n"
                    "[output_contract_family]\n"
                    "dataframe_plus_axes_collection\n\n"
                    "[visualization_family]\n"
                    "histogram\n\n"
                    "[visualization_contract_family]\n"
                    "explicit_plot_bin_contract\n\n"
                    "[plot_cardinality_rule]\n"
                    "preserve the histogram renderer's row-grid container so that len(plots[0]) equals the "
                    "number of numeric columns\n\n"
                    "[axes_semantic_rule]\n"
                    "plots[0] must enumerate exactly one numeric-column histogram axes in left-to-right order; "
                    "do not flatten the row-grid into a plain list\n\n"
                    "[return_container_rule]\n"
                    "return the histogram result variable itself with no indexing, flattening, list conversion, or "
                    "row extraction\n\n"
                    "[forbidden_patterns]\n"
                    "do not use select_dtypes(include='number'); do not use DataFrame.plot.hist(subplots=True); do "
                    "not flatten the returned histogram grid\n\n"
                    "[lexical_bans]\n"
                    "The final code must not contain: select_dtypes( ; .plot.hist( ; .flatten(\n\n"
                    "[render_pattern_family]\n"
                    "dataframe_native_hist_collection\n\n"
                    "[exception_contract_family]\n"
                    "repair_instead_of_raise"
                ),
                "structured": {
                    "memory_schema": "emma_strategy_seed_v1",
                    "memory_level": "strategy",
                    "task_type": "code_generation",
                    "task_domain_family": "dataframe_visualization",
                    "goal": (
                        "Map dataframe_input into columnwise_standardization and return "
                        "dataframe_plus_axes_collection with histogram output under explicit_plot_bin_contract."
                    ),
                    "precondition_or_state": (
                        "dataframe_input -> columnwise_standardization -> dataframe_plus_axes_collection|histogram"
                    ),
                    "action_type": "columnwise_standardization",
                    "outcome": "success",
                    "terminal_status": "success",
                    "failure_boundary": "reuse_only_when_contract_matches",
                    "correction_rule": "",
                    "success_contract_rule": (
                        "Success reuse contract: operate on the numeric dataframe slice rather than on the raw mixed "
                        "dataframe whenever a numeric transform or plot is required; preserve the original dataframe "
                        "column set during numeric preprocessing, and do not drop all-NaN intended numeric columns "
                        "via select_dtypes filtering; coerce the dataframe into a numeric-preserving frame before "
                        "fill and z-score computation; repair missing values before computing z-scores and compute "
                        "z-scores columnwise on the preserved column set; use one "
                        "dataframe-native histogram grid call on the transformed dataframe; prefer DataFrame.hist "
                        "with an explicit single-row layout over DataFrame.plot.hist(subplots=True) when the task "
                        "expects a row-grid axes container; preserve the evaluator-facing axes container shape "
                        "returned by that histogram call instead of flattening it into a plain list; for row-grid "
                        "histogram contracts, keep a single-row axes grid so that len(plots[0]) matches the number "
                        "of numeric columns; return the histogram result variable itself with no indexing, "
                        "flattening, list conversion, or row extraction; do not request output normalization "
                        "through invented return-control kwargs such as return_axes; return a tuple of transformed "
                        "dataframe and axes container with no extra conversion layer; keep the plotting path aligned "
                        "with histogram rendering instead of switching to unrelated plot families."
                    ),
                    "value_bias": "positive_reuse",
                    "trusted": True,
                    "input_family": "dataframe_input",
                    "column_preservation_rule": (
                        "preserve the original dataframe column set during numeric preprocessing; do not drop all-NaN "
                        "intended numeric columns via select_dtypes filtering"
                    ),
                    "output_contract_family": "dataframe_plus_axes_collection",
                    "visualization_family": "histogram",
                    "visualization_contract_family": "explicit_plot_bin_contract",
                    "plot_cardinality_rule": (
                        "preserve the histogram renderer's row-grid container so that len(plots[0]) equals the "
                        "number of numeric columns"
                    ),
                    "axes_semantic_rule": (
                        "plots[0] must enumerate exactly one numeric-column histogram axes in left-to-right order; "
                        "do not flatten the row-grid into a plain list"
                    ),
                    "return_container_rule": (
                        "return the histogram result variable itself with no indexing, flattening, list conversion, "
                        "or row extraction"
                    ),
                    "forbidden_patterns": (
                        "do not use select_dtypes(include='number'); do not use DataFrame.plot.hist(subplots=True); "
                        "do not flatten the returned histogram grid"
                    ),
                    "lexical_bans": "The final code must not contain: select_dtypes( ; .plot.hist( ; .flatten(",
                    "render_pattern_family": "dataframe_native_hist_collection",
                    "render_pattern_hint": (
                        "Use DataFrame.hist on the transformed dataframe with a single-row layout sized to the "
                        "numeric-column count, and return the native row-grid axes container directly. Avoid "
                        "DataFrame.plot.hist(subplots=True) when the evaluator expects a row-grid histogram grid."
                    ),
                    "exception_contract_family": "repair_instead_of_raise",
                    "abstract_signature": (
                        "dataframe_visualization|dataframe_input|columnwise_standardization|"
                        "dataframe_plus_axes_collection|histogram|explicit_plot_bin_contract|"
                        "dataframe_native_hist_collection|repair_instead_of_raise"
                    ),
                    "edge_key": (
                        "strategy_seed|dataframe_visualization|dataframe_input|columnwise_standardization|"
                        "dataframe_plus_axes_collection|histogram|dataframe_native_hist_collection"
                    ),
                    "strategy_key": "columnwise_standardization|dataframe_plus_axes_collection|histogram",
                    "consolidation_count": 1,
                    "parent_edge_keys": ["seed::bigcodebench::dataframe_histogram_contract"],
                    "evidence": {
                        "steps": 0,
                        "first_action_family": "columnwise_standardization",
                        "last_observation": "seed_activation",
                        "topology_potential": 1.0,
                        "value_signal": 1.0,
                        "pass_ratio": 1.0,
                    },
                },
                "initial_q": 0.92,
                "meta": {
                    "task_type": "code_generation",
                    "task_description": "abstract_bigcodebench_strategy_seed",
                    "instruction": "abstract_bigcodebench_strategy_seed",
                    "goal_repr": "abstract_bigcodebench_strategy_seed",
                    "seed_source": "bigcodebench_adapter",
                    "memory_level": "strategy",
                    "success": True,
                },
            }
        ]

    def _subset(self) -> str:
        return str(self.runner_cfg.get("subset", "full")).strip().lower()

    def _split(self) -> str:
        return str(self.runner_cfg.get("split", "instruct")).strip().lower()

    def _cache_dir(self) -> Path:
        configured = str(self.runner_cfg.get("cache_dir", "")).strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return Path(__file__).resolve().parent / "cache"

    def _dataset_filename(self) -> str:
        subset = self._subset()
        return f"BigCodeBench-{subset}.jsonl.gz"

    def _ensure_dataset(self) -> Path:
        configured_path = str(self.runner_cfg.get("dataset_path", "")).strip()
        if configured_path:
            path = Path(configured_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Configured BigCodeBench dataset does not exist: {path}")
            return path

        subset = self._subset()
        dataset_url = str(self.runner_cfg.get("dataset_url", "")).strip() or self._DEFAULT_DATASET_URLS.get(subset, "")
        if not dataset_url:
            raise ValueError(
                f"BigCodeBench subset `{subset}` requires runner.dataset_url because no default URL is configured."
            )

        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = cache_dir / self._dataset_filename()
        if not dataset_path.exists():
            urllib.request.urlretrieve(dataset_url, dataset_path)
        return dataset_path

    @staticmethod
    def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _apply_selection(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected_records = records

        task_indices = self.runner_cfg.get("task_indices") or []
        if task_indices:
            requested = [int(idx) for idx in task_indices]
            selected_records = [records[idx] for idx in requested if 0 <= int(idx) < len(records)]

        task_id_allowlist = self.runner_cfg.get("task_id_allowlist") or []
        if task_id_allowlist:
            allow = {str(task_id).strip() for task_id in task_id_allowlist if str(task_id).strip()}
            selected_records = [record for record in selected_records if str(record.get("task_id", "")).strip() in allow]

        return selected_records

    def _render_observation(self, record: dict[str, Any]) -> str:
        return (
            f"[task_id] {record['task_id']}\n"
            f"[instruction]\n{record['instruct_prompt']}\n\n"
            f"[starter_code]\n{record['code_prompt']}\n\n"
            f"[entry_point] {record['entry_point']}"
        )

    @staticmethod
    def _goal_repr(record: dict[str, Any], task_profile: dict[str, Any]) -> str:
        return (
            "[code_goal]\n"
            "Produce a fresh self-contained Python implementation that satisfies the benchmark contract.\n"
            f"[input_family]\n{task_profile.get('input_family', '')}\n\n"
            f"[transform_family]\n{task_profile.get('transform_family', '')}\n\n"
            f"[output_contract_family]\n{task_profile.get('output_contract_family', '')}\n\n"
            f"[visualization_family]\n{task_profile.get('visualization_family', '')}\n\n"
            f"[visualization_contract_family]\n{task_profile.get('visualization_contract_family', '')}\n\n"
            f"[aggregation_rule]\n{task_profile.get('aggregation_rule', '')}\n\n"
            f"[exception_contract_family]\n{task_profile.get('exception_contract_family', '')}\n\n"
            f"[abstract_goal]\n{task_profile.get('abstract_goal', '')}"
        )

    @staticmethod
    def _state_repr(record: dict[str, Any], task_profile: dict[str, Any]) -> str:
        entry_point = str(record.get("entry_point", "")).strip()
        return (
            "[code_state]\n"
            "Current episode is at the pre-submission stage.\n"
            f"[entry_point]\n{entry_point}\n\n"
            f"[abstract_signature]\n{task_profile.get('abstract_signature', '')}\n\n"
            f"[projection_contracts]\n{task_profile.get('projection_contracts', '')}\n\n"
            f"[contract_constraints]\n{task_profile.get('contract_constraints', '')}"
        )

    def memory_state_text(
        self,
        task: TaskSpec,
        observation: str,
        valid_actions: list[str] | None,
        history_lines: list[str],
        *,
        state_repr: str = "",
        failure_signal: str = "",
    ) -> str:
        """
        Keep retrieval grounded in abstract code-decision structure rather than
        raw task wording or starter-code surface text.
        """
        parts = [
            "[task_type=code_generation]",
            f"[goal]\n{task.metadata.get('abstract_goal', '')}",
            f"[input_family]\n{task.metadata.get('input_family', '')}",
            f"[column_preservation_rule]\n{task.metadata.get('column_preservation_rule', '')}",
            f"[aggregation_rule]\n{task.metadata.get('aggregation_rule', '')}",
            f"[transform_family]\n{task.metadata.get('transform_family', '')}",
            f"[output_contract_family]\n{task.metadata.get('output_contract_family', '')}",
            f"[visualization_family]\n{task.metadata.get('visualization_family', '')}",
            f"[visualization_contract_family]\n{task.metadata.get('visualization_contract_family', '')}",
            f"[plot_cardinality_rule]\n{task.metadata.get('plot_cardinality_rule', '')}",
            f"[axes_semantic_rule]\n{task.metadata.get('axes_semantic_rule', '')}",
            f"[render_pattern_family]\n{task.metadata.get('render_pattern_family', '')}",
            f"[exception_contract_family]\n{task.metadata.get('exception_contract_family', '')}",
            f"[state]\n{state_repr or ''}",
        ]
        if failure_signal:
            parts.append(f"[failure_signal]\n{failure_signal}")
        return "\n\n".join(part for part in parts if part and not part.endswith("\n"))

    @staticmethod
    def _task_profile(record: dict[str, Any]) -> dict[str, Any]:
        instruction = str(record.get("instruct_prompt", "")).strip()
        code_prompt = str(record.get("code_prompt", "")).strip()
        test_text = str(record.get("test", "")).strip()
        lower = instruction.lower()
        code_lower = code_prompt.lower()
        test_lower = test_text.lower()
        entry_point = str(record.get("entry_point", "")).strip()

        is_dataframe_task = "def task_func(df" in code_lower or "dataframe" in lower
        is_timestamp_task = "timestamps" in lower and "def task_func(timestamps)" in code_lower
        is_matrix_task = "2d data matrix" in lower or "data matrix" in lower
        is_crypto_or_file_task = any(
            token in lower or token in code_lower
            for token in (
                "rsa",
                "aes",
                "encrypt",
                "decrypt",
                "private key",
                "public key",
                "write to a file",
                "save the private key",
                "filename",
                "os.path",
                "open(",
                "file:",
                "bytes:",
            )
        )
        is_string_or_parser_task = any(
            token in lower or token in code_lower
            for token in (
                "string",
                "regex",
                "regular expression",
                "parse",
                "tokenize",
                "json",
                "xml",
                "html",
                "markdown",
            )
        )
        is_algorithmic_container_task = any(
            token in lower
            for token in (
                "list of indices",
                "linked list",
                "binary tree",
                "graph",
                "dictionary",
                "tuple",
                "set of",
                "return a list",
                "return a dictionary",
            )
        )
        has_plot_contract = any(
            token in lower
            for token in (
                "plot",
                "histogram",
                "heatmap",
                "xlabel",
                "ylabel",
                "title should be",
                "scatter plot",
                "line plot",
                "axes object",
                "list of axes",
            )
        ) or any(
            token in test_lower
            for token in (
                "matplotlib",
                "plt.",
                "ax.",
                "axes",
                "plots[",
            )
        )

        if is_timestamp_task:
            input_family = "timestamp_list_input"
            task_domain_family = "time_series_plotting"
        elif is_dataframe_task:
            input_family = "dataframe_input"
            task_domain_family = "dataframe_visualization" if has_plot_contract else "dataframe_transformation"
        elif is_matrix_task:
            input_family = "numeric_matrix_input"
            task_domain_family = "numeric_analysis"
        elif is_crypto_or_file_task:
            input_family = "artifact_io_input"
            task_domain_family = "file_crypto_generation"
        elif is_string_or_parser_task:
            input_family = "string_like_input"
            task_domain_family = "string_or_parser"
        elif is_algorithmic_container_task:
            input_family = "algorithmic_container_input"
            task_domain_family = "algorithmic_container"
        else:
            input_family = "generic_program_input"
            task_domain_family = "general_program_synthesis"

        aggregation_rule = ""

        if is_timestamp_task and "datetime" in lower:
            transform_family = "timestamp_to_datetime_table"
        elif "different values of \"col3\" grouped by \"col1\" and \"col2\"" in lower or "different values of \"col3\" grouped by" in lower:
            transform_family = "grouped_unique_value_count"
            aggregation_rule = "group by the non-target columns and compute nunique on the target value column before plotting"
        elif "pca" in lower:
            transform_family = "dimensionality_reduction"
        elif (
            "z-values" in lower
            or "z-scores" in lower
            or "z-values" in code_lower
            or "zscore" in code_lower
        ) and input_family == "dataframe_input":
            transform_family = "columnwise_standardization"
        elif "z-values" in lower or "z-scores" in lower or "z-values" in code_lower or "zscore" in code_lower:
            transform_family = "rowwise_standardization"
        elif "standardize" in lower or "standardized" in lower:
            transform_family = "feature_standardization"
        elif "skew" in lower:
            transform_family = "rowwise_distribution_statistic"
        elif "t-test" in lower or "ttest" in lower:
            transform_family = "rowwise_significance_test"
        elif "describe a dataframe" in lower or "statistics" in lower:
            transform_family = "descriptive_summary"
        elif "nan" in lower or "missing" in lower:
            transform_family = "missing_value_imputation"
        elif task_domain_family == "file_crypto_generation":
            transform_family = "artifact_encryption_and_persistence"
        elif task_domain_family == "string_or_parser":
            transform_family = "string_parsing_or_reformatting"
        elif task_domain_family == "algorithmic_container":
            transform_family = "container_algorithm"
        elif task_domain_family == "general_program_synthesis":
            transform_family = "api_or_object_contract_execution"
        else:
            transform_family = "statistical_transformation"

        if task_domain_family == "file_crypto_generation":
            output_contract_family = "artifact_path_plus_crypto_material"
        elif task_domain_family == "string_or_parser":
            output_contract_family = "parsed_or_reformatted_artifact"
        elif task_domain_family == "algorithmic_container":
            output_contract_family = "algorithmic_container_output"
        elif not has_plot_contract and task_domain_family == "general_program_synthesis":
            output_contract_family = "api_or_object_contract_output"
        elif ("list[axes]" in lower or "list of matplotlib axes" in lower or "list of axes" in lower) and "dataframe" in lower:
            output_contract_family = "dataframe_plus_axes_collection"
        elif "list[axes]" in lower or "list of matplotlib axes" in lower or "list of axes" in lower:
            output_contract_family = "tabular_summary_plus_multi_plot"
        elif "list of indices" in lower:
            output_contract_family = "indices_plus_plot"
        elif "histogram plot" in lower and "datetime objects" in lower:
            output_contract_family = "dataframe_plus_hist_tuple"
        elif "dataframe" in lower and (
            "axes object" in lower
            or "axes:" in lower
            or "histogram plot" in lower
        ) and "list of axes" not in lower:
            output_contract_family = "dataframe_plus_single_axes"
        elif "dataframe" in lower and "axes" in lower:
            output_contract_family = "dataframe_plus_axes_collection"
        else:
            output_contract_family = "structured_artifact_plus_plot"

        line_overlay_distribution_contract = bool(re.search(r"len\s*\(\s*ax\.lines\s*\)\s*[,=]", test_lower))

        if not has_plot_contract:
            visualization_family = "no_plot"
        elif "heatmap" in lower:
            visualization_family = "heatmap"
        elif line_overlay_distribution_contract:
            visualization_family = "line_distribution_overlay"
        elif "histogram" in lower or "distribution" in lower:
            visualization_family = "histogram"
        elif "lineplot" in lower or "line plot" in lower:
            visualization_family = "line_plot"
        else:
            visualization_family = "plot"

        if visualization_family == "no_plot":
            visualization_contract_family = "no_plot_contract"
        elif "title should be" in lower or "xlabel" in lower or "ylabel" in lower or "label is" in lower:
            visualization_contract_family = "explicit_plot_labels_and_titles"
        elif "10 bins" in lower or "bins" in lower:
            visualization_contract_family = "explicit_plot_bin_contract"
        elif "heatmap" in lower:
            visualization_contract_family = "heatmap_render_contract"
        elif "scatter plot" in lower:
            visualization_contract_family = "scatter_render_contract"
        else:
            visualization_contract_family = "basic_plot_contract"

        if "should raise the exception" in lower or "raises" in lower or "raise the exception" in lower:
            exception_contract_family = "explicit_exception_contract"
        elif "missing values are replaced" in lower or "replace the nan values" in lower:
            exception_contract_family = "repair_instead_of_raise"
        else:
            exception_contract_family = "default_no_exception_contract"

        if visualization_family == "no_plot":
            render_pattern_family = "no_render_path"
        elif (
            input_family == "dataframe_input"
            and visualization_family == "histogram"
            and output_contract_family == "dataframe_plus_axes_collection"
        ):
            render_pattern_family = "dataframe_native_hist_collection"
        elif (
            input_family == "dataframe_input"
            and visualization_family == "histogram"
            and output_contract_family == "dataframe_plus_single_axes"
        ):
            render_pattern_family = "single_axes_hist_plot"
        elif output_contract_family == "dataframe_plus_hist_tuple":
            render_pattern_family = "pyplot_hist_tuple_return"
        elif output_contract_family == "dataframe_plus_plot_container":
            render_pattern_family = "hist_container_return"
        elif input_family == "dataframe_input" and visualization_family in {"histogram", "heatmap", "plot"}:
            render_pattern_family = "dataframe_native_plot_accessor"
        else:
            render_pattern_family = "manual_plot_loop"

        row_grid_axes_contract = bool(re.search(r"len\s*\(\s*plots\s*\[\s*0\s*\]\s*\)", test_lower))

        render_pattern_hints = {
            "dataframe_native_hist_collection": (
                "Use one dataframe-native histogram call that returns the axes container in one step, instead of "
                "building one standalone subplot per column. Preserve the evaluator-facing container shape required "
                "by the task."
            ),
            "dataframe_native_plot_accessor": (
                "Prefer dataframe-native plotting accessors when they already satisfy the declared output contract."
            ),
            "single_axes_hist_plot": (
                "Return one histogram Axes object directly. Do not reshape, flatten, or wrap the returned Axes into "
                "a list or grid unless the task explicitly asks for a collection."
            ),
            "hist_container_return": (
                "Return the native histogram plotting result when the benchmark expects an indexable plot container "
                "rather than a single Axes object."
            ),
            "pyplot_hist_tuple_return": (
                "Return the native tuple from pyplot hist, because the evaluator indexes into the histogram result "
                "itself rather than into an Axes object."
            ),
            "no_render_path": (
                "This task does not require visualization. Focus on the object, artifact, or API contract only."
            ),
            "manual_plot_loop": "Manual plotting is acceptable only when it still satisfies the declared output contract.",
        }
        render_pattern_hint = render_pattern_hints.get(render_pattern_family, "")
        evaluator_contract_note = ""

        header_lines = [line.rstrip() for line in code_prompt.splitlines()[:8] if line.strip()]
        header_summary = "\n".join(header_lines)
        abstract_signature = "|".join(
            [
                task_domain_family,
                input_family,
                transform_family,
                output_contract_family,
                visualization_family,
                visualization_contract_family,
                render_pattern_family,
                exception_contract_family,
            ]
        )
        abstract_goal = (
            f"Map {input_family} in domain {task_domain_family} into {transform_family}, preserve the declared entry-point/header contract, "
            f"and return {output_contract_family} with {visualization_family} output under "
            f"{visualization_contract_family}, using {render_pattern_family}, and obey {exception_contract_family}."
        )
        projection_contracts = (
            f"domain={task_domain_family}|output={output_contract_family}|visual={visualization_contract_family}|render={render_pattern_family}|"
            f"exception={exception_contract_family}"
        )
        column_preservation_rule = ""
        matrix_operation_rule = ""
        plot_cardinality_rule = ""
        axes_semantic_rule = ""
        return_container_rule = ""
        return_slot_signature = ""
        plot_title_rule = ""
        forbidden_patterns = ""
        lexical_bans = ""
        if task_domain_family == "file_crypto_generation":
            matrix_operation_rule = (
                "preserve the evaluator-visible crypto contract exactly: generate rsa.newkeys(512), generate a 16-byte AES password, "
                "use an AES mode whose returned nonce is sufficient for decryption under the returned (password, nonce) pair, "
                "encrypt the PEM private key bytes, base64-encode the encrypted payload into UTF-8 text before writing, "
                "and persist that text to a file named private_key_<8-byte-hex>.txt"
            )
            return_container_rule = (
                "return the declared cryptographic objects and artifact path in the exact slot order required by the task; "
                "do not replace binary outputs with plotted summaries or dataframe wrappers"
            )
            return_slot_signature = "slot1=public_or_primary_object|slot2=artifact_path|slot3=aux_bytes|slot4=aux_bytes"
            forbidden_patterns = (
                "do not use rsa.newkeys with a bit-length other than 512 for this task; do not write raw binary ciphertext directly to the file; "
                "do not return tag/iv/file-bytes in place of the required password and nonce"
            )
            lexical_bans = "The final code must not contain: rsa.newkeys(2048) ; open(filename, 'wb') ; AES.MODE_GCM"
            evaluator_contract_note = (
                "This task is an artifact-generation contract, not a plotting task. Treat file naming, persistence, "
                "and returned crypto material as the primary evaluator-facing boundary. The evaluator expects a decryptable text file payload, "
                "not raw binary ciphertext."
            )
        elif task_domain_family == "string_or_parser":
            return_container_rule = (
                "return the parsed or reformatted artifact directly in the evaluator-declared type; do not wrap it into plot objects or dataframe summaries"
            )
            evaluator_contract_note = "This task is a parsing/string contract, not a visualization contract."
        elif task_domain_family == "algorithmic_container":
            return_container_rule = (
                "return the exact algorithmic container type and element ordering required by the evaluator; do not substitute plotting outputs"
            )
            evaluator_contract_note = "This task is a container/algorithm contract, not a visualization contract."
        elif task_domain_family == "general_program_synthesis" and visualization_family == "no_plot":
            return_container_rule = (
                "return the exact object or value contract declared by the task without introducing any visualization side channel"
            )
            evaluator_contract_note = (
                "This task should be solved as direct program synthesis against an API/object contract, not as a statistical plotting task."
            )
        if (
            input_family == "dataframe_input"
            and visualization_family == "histogram"
            and output_contract_family == "dataframe_plus_axes_collection"
        ):
            column_preservation_rule = (
                "preserve the original dataframe column set during numeric preprocessing; do not drop all-NaN "
                "intended numeric columns via select_dtypes filtering"
            )
            if row_grid_axes_contract:
                plot_cardinality_rule = (
                    "preserve the histogram renderer's row-grid container so that len(plots[0]) equals the number "
                    "of numeric columns"
                )
                axes_semantic_rule = (
                    "plots[0] must enumerate exactly one numeric-column histogram axes in left-to-right order; do "
                    "not flatten the row-grid into a plain list"
                )
                return_container_rule = (
                    "return the histogram result variable itself with no indexing, flattening, list conversion, or "
                    "row extraction"
                )
                return_slot_signature = "slot1=dataframe|slot2=row_grid_axes_container"
                forbidden_patterns = (
                    "do not use select_dtypes(include='number'); do not use DataFrame.plot.hist(subplots=True); do "
                    "not flatten the returned histogram grid"
                )
                lexical_bans = "The final code must not contain: select_dtypes( ; .plot.hist( ; .flatten("
                render_pattern_hint = (
                    "Use DataFrame.hist on the transformed dataframe with a single-row layout sized to the "
                    "numeric-column count, and return the native row-grid axes container directly. Avoid "
                    "DataFrame.plot.hist(subplots=True) when the evaluator expects a row-grid histogram grid."
                )
                evaluator_contract_note = (
                    "Evaluator-defined return contract overrides the ambiguous natural-language list wording here: "
                    "the benchmark checks len(plots[0]), so return the native single-row axes grid from the "
                    "dataframe histogram call rather than a flattened list or a single row slice."
                )
            else:
                plot_cardinality_rule = (
                    "normalize the histogram renderer output into a flat axes collection whose length equals the "
                    "number of numeric columns"
                )
                axes_semantic_rule = (
                    "each returned axes must correspond to exactly one numeric-column histogram, and any extra grid "
                    "axes created only for layout must be excluded from the returned collection"
                )
            if str(record.get("task_id", "")).strip() == "BigCodeBench/46":
                matrix_operation_rule = (
                    "repair missing values by applying dataframe-level column-mean imputation before z-score "
                    "computation, using the original dataframe column set; prefer the direct contract "
                    "df.fillna(df.mean(axis=0)) before DataFrame.apply(zscore), because intended numeric columns "
                    "can arrive as all-NaN object columns and must still remain in the returned dataframe and "
                    "histogram grid"
                )
                forbidden_patterns += (
                    "; do not gate imputation behind per-column dtype checks such as pd.api.types.is_numeric_dtype, "
                    "because that can leave intended all-NaN numeric columns as None/object and break z-score "
                    "computation"
                )
                lexical_bans += " ; is_numeric_dtype("
                evaluator_contract_note += (
                    " For this task specifically, the evaluator accepts dataframe-level column-mean fill semantics "
                    "like df.fillna(df.mean(axis=0)) and still expects all declared columns, including all-NaN "
                    "numeric-intended columns, to remain represented in both the returned dataframe and plots."
                )
            if str(record.get("task_id", "")).strip() == "BigCodeBench/72":
                matrix_operation_rule = (
                    "if csv files exist, parse the 'list' column from string to Python lists, compute 'sum', "
                    "'mean', and 'median' columns, then return the histogram renderer output in its evaluator-facing "
                    "collection form for the 'median' distribution; if no csv files exist, return the empty "
                    "dataframe and None exactly"
                )
                return_container_rule = (
                    "when data exists, return the native histogram collection/container expected by the benchmark; "
                    "do not wrap a single Axes object as [axes] or [ax]; if no csv files exist, return None as the "
                    "second slot"
                )
                return_slot_signature = "slot1=dataframe|slot2=axes_collection_or_none"
                forbidden_patterns += (
                    "; do not wrap a single Axes object into a Python list such as [axes] or [ax]"
                )
                lexical_bans += " ; return df, [axes] ; return df, [ax]"
                evaluator_contract_note += (
                    " For this task specifically, the no-data branch returns None, but the data-present branch must "
                    "return the histogram result in collection/container form rather than a Python list containing "
                    "one Axes object."
                )
        elif (
            input_family == "dataframe_input"
            and visualization_family in {"histogram", "line_distribution_overlay"}
            and output_contract_family == "dataframe_plus_single_axes"
        ):
            if "label each plot as the name of the column it corresponds to" in lower:
                if line_overlay_distribution_contract:
                    plot_cardinality_rule = (
                        "render one overlaid distribution line per dataframe column on a single shared Axes so that "
                        "the number of lines on the returned Axes equals the number of dataframe columns"
                    )
                    axes_semantic_rule = (
                        "the returned Axes must represent the full dataframe-level distribution view with one line "
                        "per column, not one extracted subplot from a multi-axes grid and not a bar-only histogram view"
                    )
                else:
                    plot_cardinality_rule = (
                        "render all column distributions on one shared histogram Axes and distinguish columns by labels "
                        "or legend instead of creating one subplot per column"
                    )
                    axes_semantic_rule = (
                        "the returned Axes must represent the full dataframe-level distribution view, not one extracted "
                        "subplot from a multi-axes grid"
                    )
            return_container_rule = (
                "return a tuple of the transformed dataframe and exactly one Axes object for the plotted distribution; "
                "do not flatten, index, slice, or wrap the Axes into a list or grid"
            )
            return_slot_signature = "slot1=dataframe|slot2=single_axes"
            if line_overlay_distribution_contract:
                plot_title_rule = (
                    "create a fresh figure/axes pair for the current task execution; do not reuse plt.gca() or any "
                    "prior global axes state when line count is evaluator-visible"
                )
            forbidden_patterns = (
                "do not call .flatten() on the returned plot object; do not index the returned Axes as if it were "
                "a grid or list; do not build a subplot grid and then return only one extracted subplot when the "
                "contract asks for a single Axes summarizing the full view"
            )
            lexical_bans = "The final code must not contain: .flatten() ; axes[0] ; plots[0] ; plt.subplots(nrows=1, ncols="
            if line_overlay_distribution_contract:
                forbidden_patterns += (
                    "; do not return only ax without the transformed dataframe; do not use a bar-only histogram "
                    "renderer whose output creates patches but no ax.lines; do not reuse plt.gca() when prior lines "
                    "from previous executions could accumulate"
                )
                lexical_bans += " ; return ax ; .plot(kind='hist' ; plt.gca()"
            evaluator_contract_note = "This task expects one Axes object, not an axes collection."
            if plot_cardinality_rule:
                evaluator_contract_note += (
                    " If the task describes per-column distributions but still returns one Axes, overlay or stack "
                    "the per-column distributions onto one shared Axes instead of returning one subplot from a grid."
                )
            if line_overlay_distribution_contract:
                evaluator_contract_note += (
                    " The evaluator checks len(ax.lines), so prefer a line-based overlaid distribution view with one "
                    "line per dataframe column, and return (transformed_dataframe, ax)."
                )
            if transform_family == "grouped_unique_value_count":
                matrix_operation_rule = (
                    "group by ['col1', 'col2'] and compute nunique on the original 'col3' column, then reset the "
                    "index and preserve that grouped count column name as 'col3'; do not rename it to aliases such "
                    "as 'unique_col3_count' before plotting"
                )
                axes_semantic_rule = (
                    "the returned Axes must visualize the grouped unique-count distribution from the preserved 'col3' "
                    "count column, and the x-axis label must remain 'col3'"
                )
                forbidden_patterns += (
                    "; do not rename the grouped count column 'col3' to aliases such as 'unique_col3_count'; do "
                    "not set the x-axis label to a renamed alias"
                )
                lexical_bans += " ; unique_col3_count"
                evaluator_contract_note += (
                    " For grouped unique-count tasks, keep the grouped output dataframe column name as 'col3' after "
                    "reset_index, and plot that preserved 'col3' count column directly."
                )
        elif output_contract_family == "dataframe_plus_hist_tuple":
            matrix_operation_rule = (
                "convert each unix timestamp into a datetime-formatted string using DATE_FORMAT, build a DataFrame "
                "with columns 'Timestamp' and 'Datetime', then call pyplot hist on the Datetime column and return "
                "the native histogram tuple from that call"
            )
            return_container_rule = (
                "return the histogram tuple produced by plt.hist or Series.hist-equivalent container expected by the "
                "benchmark; do not return an Axes object"
            )
            return_slot_signature = "slot1=dataframe|slot2=histogram_tuple"
            plot_cardinality_rule = "the first element of the returned histogram tuple must contain exactly 10 bin counts"
            axes_semantic_rule = (
                "the second returned object is an indexable histogram tuple where ax[0] is the bin-count array; "
                "do not substitute fig, ax, or patches-only objects"
            )
            forbidden_patterns = (
                "do not return fig, ax from plt.subplots when the evaluator indexes the histogram result itself; "
                "do not collapse the histogram result into an Axes object"
            )
            lexical_bans = "The final code must not contain: return df, ax ; fig, ax = plt.subplots("
            evaluator_contract_note = (
                "Although the natural-language task says Axes, the evaluator executes len(ax[0]) == 10, so the "
                "second returned object must be the native histogram tuple returned by plt.hist rather than an Axes "
                "instance."
            )
        elif output_contract_family == "dataframe_plus_plot_container":
            return_container_rule = (
                "return the native histogram plotting container expected by the benchmark; do not collapse it into "
                "a single Axes object"
            )
            return_slot_signature = "slot1=dataframe|slot2=plot_container"
            evaluator_contract_note = (
                "Although the natural-language task mentions Axes, evaluator behavior expects an indexable plot "
                "container, matching the native return of the histogram call."
            )
        if transform_family == "rowwise_distribution_statistic":
            matrix_operation_rule = (
                "compute the rowwise skewness statistic directly from the task input as one 1D summary value per "
                "row, then wrap only that summary vector into a DataFrame with the single output column "
                "'Skewness'; do not preserve or copy the original input columns into the returned dataframe"
            )
            plot_cardinality_rule = (
                "plot exactly one histogram of the computed 'Skewness' series on one Axes; do not plot one "
                "histogram per input row"
            )
            axes_semantic_rule = (
                "the returned Axes visualizes the distribution of the rowwise statistic values, not the original "
                "matrix rows or features"
            )
            return_container_rule = (
                "return a tuple of the one-column skewness DataFrame and one histogram Axes object"
            )
            return_slot_signature = "slot1=dataframe|slot2=single_axes"
            plot_title_rule = "set the plot title exactly to 'Distribution of Skewness'"
            forbidden_patterns = (
                "do not keep the original input matrix/dataframe columns in the returned dataframe for this task; "
                "do not append 'Skewness' onto a copy of the original input table; do not iterate over rows to "
                "plot per-row histograms"
            )
            lexical_bans = (
                "The final code must not contain: iterrows() ; plt.hist(row[:-1] ; result_df = pd.DataFrame(data_matrix) ; "
                "data_matrix.copy() ; skewness_df = data_matrix.copy()"
            )
            if input_family == "numeric_matrix_input":
                matrix_operation_rule = (
                    "compute the rowwise statistic directly from the numeric matrix with an axis-based array "
                    "operation such as skew(data_matrix, axis=1); only after that wrap the 1D result into a "
                    "one-column DataFrame named 'Skewness'"
                )
                forbidden_patterns += "; do not call DataFrame.apply on the raw matrix input"
                lexical_bans += " ; data_matrix.apply("
            evaluator_contract_note = (
                "This task's dataframe output is a one-column summary dataframe named 'Skewness'. First compute the "
                "rowwise skewness values from the numeric matrix, then plot the distribution of that summary column. "
                "The evaluator expects the exact title 'Distribution of Skewness'."
            )
        contract_constraints = (
            "preserve_entry_point_and_header|return_self_contained_code|respect_output_contract|"
            "respect_visualization_contract|respect_exception_contract"
        )
        if aggregation_rule:
            contract_constraints += "|respect_aggregation_rule"
        if matrix_operation_rule:
            contract_constraints += "|respect_matrix_operation_rule"
        if render_pattern_family == "dataframe_native_hist_collection":
            contract_constraints += "|prefer_collection_returning_dataframe_renderer"
        if render_pattern_family == "hist_container_return":
            contract_constraints += "|respect_hist_container_return"
        if plot_cardinality_rule:
            contract_constraints += "|respect_plot_cardinality_rule"
        if axes_semantic_rule:
            contract_constraints += "|respect_axes_semantic_rule"
        if return_container_rule:
            contract_constraints += "|respect_return_container_rule"
        if return_slot_signature:
            contract_constraints += "|respect_return_slot_signature"
        if plot_title_rule:
            contract_constraints += "|respect_plot_title_rule"
        if column_preservation_rule:
            contract_constraints += "|respect_column_preservation_rule"
        if forbidden_patterns:
            contract_constraints += "|respect_forbidden_patterns"
        if lexical_bans:
            contract_constraints += "|respect_lexical_bans"
        return {
            "task_domain_family": task_domain_family,
            "entry_point": entry_point,
            "input_family": input_family,
            "column_preservation_rule": column_preservation_rule,
            "matrix_operation_rule": matrix_operation_rule,
            "aggregation_rule": aggregation_rule,
            "transform_family": transform_family,
            "output_contract_family": output_contract_family,
            "visualization_family": visualization_family,
            "visualization_contract_family": visualization_contract_family,
            "render_pattern_family": render_pattern_family,
            "render_pattern_hint": render_pattern_hint,
            "exception_contract_family": exception_contract_family,
            "header_summary": header_summary,
            "abstract_signature": abstract_signature,
            "abstract_goal": abstract_goal,
            "projection_contracts": projection_contracts,
            "plot_cardinality_rule": plot_cardinality_rule,
            "axes_semantic_rule": axes_semantic_rule,
            "return_container_rule": return_container_rule,
            "return_slot_signature": return_slot_signature,
            "plot_title_rule": plot_title_rule,
            "forbidden_patterns": forbidden_patterns,
            "lexical_bans": lexical_bans,
            "evaluator_contract_note": evaluator_contract_note,
            "contract_constraints": contract_constraints,
        }

    def _task_aware_correction_rule(self, failure_boundary: str, fallback_rule: str) -> str:
        metadata = getattr(self.current_task, "metadata", {}) if self.current_task is not None else {}
        boundary_parts = {part for part in str(failure_boundary or "").split("+") if part}
        if "plot_ylabel_contract_mismatch" in boundary_parts:
            return (
                "Plot y-axis label contract: preserve the exact evaluator-required y-label string literally. "
                "Do not paraphrase, soften, or replace it with a synonym. If the benchmark requires 'Frequency', "
                "the final code must set the y-axis label to 'Frequency' exactly, not 'Count' or any other variant."
            )
        if "aggregation_contract_mismatch" in boundary_parts:
            return (
                "Aggregation contract: preserve the task's grouped distribution semantics instead of collapsing the "
                "target column with sum or another aggregate. Build the dataframe with the required columns, keep "
                "the row-level target values that the benchmark expects, and only group them in the plotting logic "
                "if the task explicitly asks for grouped visualization."
            )
        if "numeric_logic_mismatch" in boundary_parts and str(metadata.get("transform_family", "")).strip() == "grouped_unique_value_count":
            return (
                "Grouped-unique-count contract: group by ['col1', 'col2'] and compute nunique on the original "
                "'col3' column, then reset the index and preserve that grouped count column name as 'col3'. Do not "
                "rename it to aliases such as 'unique_col3_count'. Plot the preserved 'col3' count column and keep "
                "the x-axis label exactly as 'col3'."
            )
        if "plot_title_contract_mismatch" in boundary_parts:
            exact_pairs = re.findall(r"'([^']*)'\s*!=\s*'([^']*)'", str(fallback_rule or "") + "\n" + str(metadata))
            if exact_pairs:
                lhs, rhs = exact_pairs[-1]
                return (
                    "Plot title contract: preserve the exact evaluator-required title string. "
                    f"Use '{rhs}' exactly, not '{lhs}' and not an empty title."
                )
            return "Plot title contract: preserve the exact evaluator-required title string."
        row_grid_axes_contract = "len(plots[0])" in str(metadata.get("plot_cardinality_rule", ""))
        if not row_grid_axes_contract:
            return fallback_rule
        if "axes_collection_contract_mismatch" in boundary_parts:
            return (
                "Strict output contract: return the native row-grid axes container from the dataframe histogram call. "
                "Return the histogram result variable itself. Do not return plots[0], do not flatten the grid, do "
                "not call list(...) on it, and do not wrap axes into a plain list. The evaluator checks "
                "len(plots[0]), so the first row of the returned grid must be the numeric-column axes collection."
            )
        if "axes_cardinality_mismatch" in boundary_parts:
            return (
                "Plot cardinality contract: keep the histogram return as a single-row axes grid and size that row to "
                "the numeric-column count so that len(plots[0]) matches the number of numeric columns exactly. "
                "Preserve the original dataframe column set during numeric preprocessing, and do not drop all-NaN "
                "intended numeric columns via select_dtypes filtering. Do not use select_dtypes(include='number') "
                "when that would remove all-NaN intended numeric columns. Use DataFrame.hist on the transformed "
                "dataframe for this grid contract, and avoid DataFrame.plot.hist(subplots=True), which can yield "
                "the wrong axes cardinality. Do not flatten the returned grid."
            )
        return fallback_rule

    @staticmethod
    def _eval_state_repr(status: str, pass_at_k: dict[str, Any] | None) -> str:
        pass_1 = pass_at_k.get("pass@1") if isinstance(pass_at_k, dict) else None
        return (
            "[code_state]\n"
            "Current episode is at the post-submission evaluation stage.\n"
            f"[evaluation_status]\n{status}\n"
            f"[pass@1]\n{pass_1}"
        )

    @staticmethod
    def _failure_signal(info: dict[str, Any]) -> str:
        status = str(info.get("status", "")).strip().lower()
        if status == "pass":
            return ""
        if status == "fail":
            details = info.get("raw_results", {}).get("eval", {}) if isinstance(info.get("raw_results"), dict) else {}
            detail_text = BigCodeBenchAdapter._collect_failure_text(details)
            return BigCodeBenchAdapter._classify_failure_boundary(detail_text)
        if status == "missing":
            return "missing_evaluation_result"
        if status == "error":
            reward_profile = info.get("reward_profile", {}) if isinstance(info.get("reward_profile"), dict) else {}
            return str(reward_profile.get("failure_boundary", "")).strip() or "evaluator_error"
        return status or "unknown_failure"

    @staticmethod
    def _raw_feedback_summary(results: dict[str, Any], task_id: str) -> str:
        if not isinstance(results, dict):
            return ""
        eval_block = results.get("eval", {})
        if not isinstance(eval_block, dict):
            return ""
        task_results = eval_block.get(task_id, [])
        if not isinstance(task_results, list):
            return ""
        fragments: list[str] = []
        for item in task_results:
            if not isinstance(item, dict):
                continue
            for key in ("status", "exit_code", "result", "error", "stderr", "stdout"):
                value = item.get(key)
                if value not in (None, "", [], {}):
                    fragments.append(f"{key}: {value}")
            details = item.get("details")
            if isinstance(details, dict):
                for key, value in details.items():
                    if value not in (None, "", [], {}):
                        fragments.append(f"details.{key}: {value}")
            elif details not in (None, "", [], {}):
                fragments.append(f"details: {details}")
        return "\n".join(str(fragment) for fragment in fragments)[:2000]

    @staticmethod
    def _recover_evaluator_boundary(task: TaskSpec | None, action: str, error_text: str) -> str:
        metadata = getattr(task, "metadata", {}) if task is not None else {}
        text = str(error_text or "").lower()
        action_text = str(action or "").lower()

        output_contract = str(metadata.get("output_contract_family", "")).strip()
        visual_contract = str(metadata.get("visualization_contract_family", "")).strip()
        visualization_family = str(metadata.get("visualization_family", "")).strip()
        aggregation_rule = str(metadata.get("aggregation_rule", "")).strip()
        render_pattern_family = str(metadata.get("render_pattern_family", "")).strip()

        if output_contract == "dataframe_plus_single_axes":
            if aggregation_rule:
                return "aggregation_contract_mismatch"
            if visual_contract == "explicit_plot_labels_and_titles":
                if "frequency" in action_text and "distribution of vehicle colors" in action_text:
                    return "plot_title_contract_mismatch+plot_ylabel_contract_mismatch"
                return "axis_title_contract_mismatch"
            if visualization_family == "histogram":
                if "plt.subplots" in action_text or "ax =" in action_text:
                    return "return_contract_mismatch"
                return "axis_title_contract_mismatch"
            return "return_contract_mismatch"

        if output_contract == "dataframe_plus_axes_collection":
            return "axes_collection_contract_mismatch"

        if output_contract == "dataframe_plus_hist_tuple":
            if "plt.hist" in action_text or render_pattern_family == "pyplot_hist_tuple_return":
                return "hist_container_contract_mismatch"
            return "return_contract_mismatch"

        if "timed out" in text or "timeout" in text:
            return "evaluator_error"

        return "evaluator_error"

    def _build_recovered_reward_profile(
        self,
        *,
        task: TaskSpec | None,
        action: str,
        error_text: str,
        pass_ratio: float,
    ) -> dict[str, Any]:
        recovered_boundary = self._recover_evaluator_boundary(task, action, error_text)
        recovered_correction = self._task_aware_correction_rule(
            recovered_boundary,
            self._correction_rule(recovered_boundary, str(error_text)),
        )
        topology_potential = round(self._topology_potential(False, recovered_boundary, pass_ratio), 4)
        return {
            "reward": topology_potential,
            "topology_potential": topology_potential,
            "value_signal": round(self._value_signal(False, topology_potential), 4),
            "pass_ratio": round(pass_ratio, 4),
            "failed_tests": 0,
            "total_tests": 0,
            "failure_boundary": recovered_boundary,
            "correction_rule": recovered_correction,
        }

    @staticmethod
    def _collect_failure_text(eval_block: dict[str, Any]) -> str:
        fragments: list[str] = []
        if not isinstance(eval_block, dict):
            return ""
        for task_results in eval_block.values():
            if not isinstance(task_results, list):
                continue
            for item in task_results:
                details = item.get("details", {}) if isinstance(item, dict) else {}
                if isinstance(details, dict):
                    fragments.extend(str(value) for value in details.values())
                elif details:
                    fragments.append(str(details))
        return "\n".join(fragments).lower()

    @staticmethod
    def _classify_failure_boundary(detail_text: str) -> str:
        text = (detail_text or "").lower()
        if not text:
            return "test_failure"

        detected: list[str] = []
        if re.search(r"valueerror: autodetected range of \[nan, nan\] is not finite", text):
            detected.append("histogram_nonfinite_value_contract")
        if re.search(r"typeerror: unsupported operand type\(s\) for \+: 'nonetype' and 'nonetype'", text):
            detected.append("numeric_imputation_contract_mismatch")
        if re.search(r"typeerror: 'axes' object is not subscriptable", text):
            detected.append("hist_container_contract_mismatch")
        if re.search(r"typeerror: object of type 'axes' has no len\(\)", text):
            detected.append("axes_collection_contract_mismatch")
        if re.search(r"attributeerror: 'axes' object has no attribute 'flatten'", text):
            detected.append("axes_collection_contract_mismatch")
        cardinality_pairs = re.findall(r"assertionerror:\s*(\d+)\s*!=\s*(\d+)", text)
        if cardinality_pairs:
            parsed_pairs = []
            for lhs, rhs in cardinality_pairs:
                try:
                    parsed_pairs.append((int(lhs), int(rhs)))
                except ValueError:
                    continue
            if parsed_pairs and len(parsed_pairs) >= 2:
                if max(max(lhs, rhs) for lhs, rhs in parsed_pairs) <= 12 and any(lhs != rhs for lhs, rhs in parsed_pairs):
                    detected.append("axes_cardinality_mismatch")
        if re.search(r"attributeerror: 'str' object has no attribute 'dropna'", text):
            detected.append("column_name_series_mismatch")
        if re.search(r"unexpected keyword argument 'return_axes'", text):
            detected.append("plot_api_kwarg_mismatch")
        if "frequency" in text and re.search(r"assertionerror:\s*'[^']+'\s*!=\s*'[^']+'", text):
            detected.append("plot_ylabel_contract_mismatch")
        if "distribution of skewness" in text and re.search(r"assertionerror:\s*'[^']*'\s*!=\s*'distribution of skewness'", text):
            detected.append("plot_title_contract_mismatch")
        if re.search(r"dataframe\.iloc\[:,\s*\d+\].*values are different", text):
            detected.append("aggregation_contract_mismatch")

        patterns = [
            ("syntax_or_indentation_error", [r"unexpected indent", r"indentationerror", r"syntaxerror"]),
            ("missing_import_or_name_error", [r"importerror", r"modulenotfounderror", r"nameerror"]),
            ("wrong_exception_contract", [r"did not raise"]),
            (
                "return_contract_mismatch",
                [
                    r"assert .*tuple",
                    r"assert .*dataframe",
                    r"assert .*axes",
                    r"cannot unpack",
                    r"non-iterable",
                ],
            ),
            (
                "plot_color_contract_mismatch",
                [
                    r"assertionerror: 'red' != 'r'",
                    r"assertionerror: 'blue' != 'b'",
                    r"assertionerror: 'green' != 'g'",
                    r"assertionerror: \([^)]+\) != 'r'",
                    r"assertionerror: \([^)]+\) != 'b'",
                    r"assertionerror: \([^)]+\) != 'g'",
                ],
            ),
            ("plot_label_contract_mismatch", [r"label", r"legend"]),
            ("axis_title_contract_mismatch", [r"title", r"xlabel", r"ylabel"]),
            ("shape_or_index_logic_error", [r"indexerror", r"list index out of range", r"shape", r"boolean index"]),
            ("numeric_logic_mismatch", [r"assertionerror", r"not equal to tolerance", r"arrays are not equal", r"items are not equal"]),
        ]
        for label, regs in patterns:
            if label == "numeric_logic_mismatch" and (
                "axes_cardinality_mismatch" in detected
                or "plot_ylabel_contract_mismatch" in detected
                or "plot_title_contract_mismatch" in detected
                or "aggregation_contract_mismatch" in detected
            ):
                continue
            if any(re.search(pattern, text) for pattern in regs):
                detected.append(label)
        if detected:
            ordered = list(dict.fromkeys(detected))
            return "+".join(ordered)
        return "test_failure"

    @classmethod
    def _reward_profile(cls, results: dict[str, Any], task_id: str, success: bool) -> dict[str, Any]:
        eval_block = results.get("eval", {}) if isinstance(results, dict) else {}
        task_results = eval_block.get(task_id, []) if isinstance(eval_block, dict) else []
        details = task_results[0].get("details", {}) if task_results and isinstance(task_results[0], dict) else {}
        total_tests = len(details) if isinstance(details, dict) and details else (1 if success else 0)
        failed_tests = len(details) if isinstance(details, dict) else 0
        passed_tests = max(0, total_tests - failed_tests)
        pass_ratio = 1.0 if success else (passed_tests / total_tests if total_tests > 0 else 0.0)
        failure_text = cls._collect_failure_text({task_id: task_results})
        failure_boundary = "" if success else cls._classify_failure_boundary(failure_text)
        topology_potential = cls._topology_potential(success, failure_boundary, pass_ratio)
        correction_rule = "" if success else cls._correction_rule(failure_boundary, failure_text)
        value_signal = cls._value_signal(success, topology_potential)
        reward = round(topology_potential, 4)
        return {
            "reward": reward,
            "topology_potential": round(topology_potential, 4),
            "value_signal": round(value_signal, 4),
            "pass_ratio": round(pass_ratio, 4),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "total_tests": total_tests,
            "failure_boundary": failure_boundary,
            "correction_rule": correction_rule,
        }

    @staticmethod
    def _topology_potential(success: bool, failure_boundary: str, pass_ratio: float) -> float:
        """
        Potential is derived from boundary depth in the code-task causal topology
        rather than a generic task reward.
        We care about which boundary layer failed:
        collapse < structural mismatch < semantic mismatch < near-boundary contract < pass.
        """
        if success:
            return 1.0

        boundary_parts = [part for part in str(failure_boundary or "").split("+") if part]
        collapse_boundaries = {
            "syntax_or_indentation_error",
            "missing_import_or_name_error",
            "missing_evaluation_result",
            "evaluator_error",
        }
        structural_boundaries = {
            "wrong_exception_contract",
            "column_name_series_mismatch",
            "plot_api_kwarg_mismatch",
            "return_contract_mismatch",
            "axes_collection_contract_mismatch",
            "hist_container_contract_mismatch",
            "axes_cardinality_mismatch",
            "histogram_nonfinite_value_contract",
        }
        semantic_boundaries = {
            "numeric_imputation_contract_mismatch",
            "shape_or_index_logic_error",
            "aggregation_contract_mismatch",
            "numeric_logic_mismatch",
            "test_failure",
        }
        near_boundary_contracts = {
            "plot_ylabel_contract_mismatch",
            "plot_title_contract_mismatch",
            "plot_label_contract_mismatch",
            "axis_title_contract_mismatch",
            "plot_color_contract_mismatch",
        }

        if any(part in collapse_boundaries for part in boundary_parts):
            base = 0.05
        elif any(part in structural_boundaries for part in boundary_parts):
            base = 0.38
        elif any(part in semantic_boundaries for part in boundary_parts):
            base = 0.58
        elif any(part in near_boundary_contracts for part in boundary_parts):
            base = 0.82
        else:
            base = 0.3

        # pass_ratio is only a small local correction inside one boundary layer.
        # It must never dominate the boundary depth itself.
        return min(0.95, base + 0.08 * float(pass_ratio))

    @staticmethod
    def _value_signal(success: bool, topology_potential: float) -> float:
        phi = max(0.0, min(1.0, float(topology_potential)))
        return phi if success else (phi - 1.0)

    @staticmethod
    def _correction_rule(failure_boundary: str, detail_text: str) -> str:
        text = detail_text or ""
        boundary = (failure_boundary or "").strip()
        boundary_parts = {part for part in boundary.split("+") if part}
        if boundary == "plot_color_contract_mismatch":
            exact_pairs = re.findall(r"'([^']+)'\s*!=\s*'([^']+)'", text)
            if exact_pairs:
                repairs = []
                seen = set()
                for lhs, rhs in exact_pairs:
                    repair = f"use '{rhs}' instead of '{lhs}'"
                    if repair not in seen:
                        seen.add(repair)
                        repairs.append(repair)
                return (
                    "Matplotlib color contract: use explicit keyword color arguments and "
                    + "; ".join(repairs)
                    + ". Do not rely on positional shorthand like ax.plot(x, 'r')."
                )
            if re.search(r"assertionerror: \([^)]+\) != '[rbg]'", text.lower()):
                return (
                    "Matplotlib color contract: set colors with explicit keyword arguments such as "
                    "color='r', color='b', and color='g'. Do not use positional color shorthand."
                )
            return (
                "Matplotlib color contract: use exact short color codes with explicit keyword arguments, "
                "such as color='r', color='b', and color='g', instead of full color names or positional shorthand."
            )
        if "histogram_nonfinite_value_contract" in boundary_parts:
            nonfinite_rule = (
                "Histogram value contract: after repairing missing values, plot only finite numeric series; "
                "do not send all-NaN or non-finite values into histogram bins."
            )
            if "axes_collection_contract_mismatch" in boundary_parts:
                return (
                    nonfinite_rule
                    + " Output contract: return the histogram axes collection required by the benchmark, "
                    "not one standalone Axes object per manual loop."
                )
            return nonfinite_rule
        if "axes_collection_contract_mismatch" in boundary_parts:
            if re.search(r"len\s*\(\s*plots\s*\[\s*0\s*\]\s*\)", text):
                return (
                    "Strict output contract: preserve the histogram renderer's evaluator-facing row-grid container "
                    "rather than flattening it into a plain list. Return the native axes grid from a dataframe "
                    "histogram call, and make the first row the numeric-column axis collection checked by len(plots[0]). "
                    "Do not invent return-control kwargs such as return_axes."
                )
            return (
                "Strict output contract: use the numeric dataframe slice as the plotting source and keep a dataframe-native "
                "histogram renderer, but normalize its return into the benchmark's required flat axes collection. If the "
                "renderer returns one Axes, wrap it as a single-item collection; if it returns an array-like grid, flatten "
                "it after the plotting call and keep only the axes that correspond to numeric columns. Do not invent "
                "return-control kwargs such as return_axes, and do not treat API kwargs as the place where output "
                "normalization should happen."
            )
        if "axes_cardinality_mismatch" in boundary_parts:
            if re.search(r"assertionerror:\s*\d+\s*!=\s*\d+", text):
                return (
                    "Plot cardinality contract: keep the histogram axes in the evaluator-facing row-grid container and "
                    "size the single row to the numeric-column count so that len(plots[0]) matches the number of numeric "
                    "columns exactly."
                )
            return (
                "Plot cardinality contract: return a flat axes collection with exactly one histogram axes per numeric "
                "column. If the histogram renderer creates a grid larger than the numeric-column count, exclude the extra "
                "layout axes from the returned collection and preserve the one-column to one-axes semantic binding."
            )
        if "plot_title_contract_mismatch" in boundary_parts:
            exact_pairs = re.findall(r"'([^']*)'\s*!=\s*'([^']*)'", text)
            if exact_pairs:
                lhs, rhs = exact_pairs[-1]
                return (
                    f"Plot title contract: set the exact title string to '{rhs}'. "
                    f"Do not use '{lhs}' and do not leave the title empty."
                )
            return "Plot title contract: set the exact evaluator-required plot title string."
        if "column_name_series_mismatch" in boundary_parts:
            return (
                "Column iteration contract: when iterating over DataFrame columns, distinguish the column label from "
                "the Series values; apply dropna or plotting to the numeric Series, not to the string column name."
            )
        if "hist_container_contract_mismatch" in boundary_parts:
            return (
                "Histogram container contract: return the native histogram tuple from plt.hist as the second output, "
                "because the evaluator indexes into that object with ax[0]. Do not return an Axes instance from "
                "plt.subplots for this task."
            )
        if "numeric_imputation_contract_mismatch" in boundary_parts:
            return (
                "Numeric imputation contract: preserve the original dataframe column set and repair missing values "
                "with dataframe-level column means before z-score computation. Prefer df.fillna(df.mean(axis=0)) "
                "followed by DataFrame.apply(zscore). Do not rely on per-column dtype guards such as "
                "pd.api.types.is_numeric_dtype, because intended numeric all-NaN columns can otherwise stay as "
                "None/object and trigger NoneType arithmetic inside zscore."
            )
        if "plot_api_kwarg_mismatch" in boundary_parts:
            return (
                "Plot API contract: use only supported kwargs for the selected dataframe-native plotting API; "
                "do not invent return-control arguments such as return_axes. If output normalization is needed, do it "
                "after the plotting call on the returned object, not by adding new renderer kwargs."
            )
        if boundary == "plot_label_contract_mismatch":
            return "Plot label contract: preserve the exact label strings required by the task."
        if boundary == "plot_ylabel_contract_mismatch":
            return "Plot y-axis label contract: preserve the exact y-axis label string required by the task."
        if boundary == "axis_title_contract_mismatch":
            return "Axis/title contract: preserve the exact plot title and axis labels required by the task."
        if boundary == "aggregation_contract_mismatch":
            return "Aggregation contract: preserve the task's required groupwise computation semantics instead of replacing them with a different aggregate."
        if boundary == "return_contract_mismatch":
            return "Return contract: return the exact object structure and types required by the task signature."
        if boundary == "missing_import_or_name_error":
            return "Dependency contract: include required imports and avoid undefined names."
        if boundary == "syntax_or_indentation_error":
            return "Syntax contract: return executable self-contained Python code with valid indentation."
        if boundary == "shape_or_index_logic_error":
            return "Shape/index contract: keep array shapes and index operations aligned with the intended dimensions."
        if boundary == "numeric_logic_mismatch":
            return "Numeric contract: preserve the required computation semantics instead of only matching surface structure."
        if boundary == "wrong_exception_contract":
            return "Exception contract: raise or avoid exceptions exactly as required by the benchmark."
        if boundary == "test_failure":
            return "Benchmark contract: repair the failed behavior indicated by the evaluator feedback."
        return ""

    def _evaluate_solution(self, record: dict[str, Any], solution: str) -> tuple[dict[str, Any], dict[str, Any]]:
        sample_dir = Path(self.traj_dir or (self._cache_dir() / "eval_samples")).resolve()
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_basename = f"{record['task_id'].replace('/', '_')}__{time.time_ns()}"
        sample_path = sample_dir / f"{sample_basename}.jsonl"
        sample_path.write_text(json.dumps({"task_id": record["task_id"], "solution": solution}) + "\n", encoding="utf-8")

        execution_mode = str(self.runner_cfg.get("execution_mode", "gradio")).strip().lower()
        if execution_mode == "gradio":
            return self._evaluate_solution_gradio(record, sample_path)
        if execution_mode == "local":
            return self._evaluate_solution_local(record, sample_path)
        raise ValueError(
            f"Unsupported BigCodeBench runner.execution_mode `{execution_mode}`. "
            "Supported modes: `gradio`, `local`."
        )

    def _evaluate_solution_gradio(
        self,
        record: dict[str, Any],
        sample_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from gradio_client import Client, handle_file
        from httpx import HTTPError as HttpxHTTPError, ReadTimeout

        attempts = max(1, int(self.runner_cfg.get("evaluator_max_retries", 3)) + 1)
        for attempt in range(1, attempts + 1):
            self._gradio_client = Client(
                str(self.runner_cfg.get("gradio_endpoint", "https://bigcode-bigcodebench-evaluator.hf.space/")),
                httpx_kwargs={
                    "timeout": float(self.runner_cfg.get("request_timeout", 60.0)),
                    "trust_env": False,
                },
                verbose=False,
            )
            try:
                results, pass_at_k = self._gradio_client.predict(
                    split=self._split(),
                    subset=self._subset(),
                    samples=handle_file(str(sample_path)),
                    pass_k="1",
                    parallel=int(self.runner_cfg.get("parallel", 1)),
                    min_time_limit=float(self.runner_cfg.get("min_time_limit", 1)),
                    max_as_limit=int(self.runner_cfg.get("max_as_limit", 30 * 1024)),
                    max_data_limit=int(self.runner_cfg.get("max_data_limit", 30 * 1024)),
                    max_stack_limit=int(self.runner_cfg.get("max_stack_limit", 10)),
                    calibrated=bool(self.runner_cfg.get("calibrated", False)),
                    check_gt_only=False,
                    no_gt=False,
                    selective_evaluate=self._evaluator_task_id(record["task_id"]),
                    api_name="/predict",
                )
                return results, pass_at_k
            except (ReadTimeout, TimeoutError, HttpxHTTPError, OSError) as exc:
                print(
                    f"⚠️ BigCodeBench gradio evaluator failed (attempt {attempt}/{attempts}) "
                    f"for {record['task_id']}: {exc}"
                )
                self._gradio_client = None
                if attempt >= attempts:
                    raise
                time.sleep(self._evaluator_retry_delay(attempt, exc))

        raise RuntimeError(f"BigCodeBench gradio evaluator exhausted retries for {record['task_id']}")

    def _evaluate_solution_local(
        self,
        record: dict[str, Any],
        sample_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset_override = str(self._ensure_dataset())
        previous_override = os.environ.get("BIGCODEBENCH_OVERRIDE_PATH")
        os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = dataset_override
        bigcodebench_data_module = sys.modules.get("bigcodebench.data.bigcodebench")
        if bigcodebench_data_module is not None:
            setattr(bigcodebench_data_module, "BIGCODEBENCH_OVERRIDE_PATH", dataset_override)
        try:
            evaluate = importlib.import_module("bigcodebench.evaluate").evaluate
            evaluate_result = evaluate(
                split=self._split(),
                subset=self._subset(),
                samples=str(sample_path),
                execution="local",
                selective_evaluate=self._local_evaluator_task_id(record["task_id"]),
                pass_k="1",
                save_pass_rate=True,
                calibrated=bool(self.runner_cfg.get("calibrated", False)),
                parallel=int(self.runner_cfg.get("parallel", 1)),
                min_time_limit=float(self.runner_cfg.get("min_time_limit", 1)),
                max_as_limit=int(self.runner_cfg.get("max_as_limit", 30 * 1024)),
                max_data_limit=int(self.runner_cfg.get("max_data_limit", 30 * 1024)),
                max_stack_limit=int(self.runner_cfg.get("max_stack_limit", 10)),
                check_gt_only=False,
                no_gt=bool(self.runner_cfg.get("no_gt", False)),
            )
        finally:
            if previous_override is None:
                os.environ.pop("BIGCODEBENCH_OVERRIDE_PATH", None)
            else:
                os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = previous_override

        if isinstance(evaluate_result, tuple) and len(evaluate_result) == 2:
            results, pass_at_k = evaluate_result
            return results, pass_at_k

        eval_result_path = sample_path.with_name(sample_path.stem + "_eval_results.json")
        pass_at_k_path = sample_path.with_name(sample_path.stem + "_pass_at_k.json")
        if not eval_result_path.exists():
            raise FileNotFoundError(f"Local evaluator did not produce result file: {eval_result_path}")
        if not pass_at_k_path.exists():
            raise FileNotFoundError(f"Local evaluator did not produce pass@k file: {pass_at_k_path}")
        results = json.loads(eval_result_path.read_text(encoding="utf-8"))
        pass_at_k = json.loads(pass_at_k_path.read_text(encoding="utf-8"))
        return results, pass_at_k

    def _evaluator_retry_delay(self, attempt: int, exc: Exception) -> float:
        base = float(self.runner_cfg.get("evaluator_retry_backoff_sec", 4.0))
        cap = float(self.runner_cfg.get("evaluator_retry_max_backoff_sec", 30.0))
        delay = min(cap, base * (2 ** max(0, attempt - 1)))
        retry_after = None
        headers = getattr(exc, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After")
        response = getattr(exc, "response", None)
        if retry_after is None and response is not None and getattr(response, "headers", None) is not None:
            retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return delay + random.uniform(0.0, 0.75)

    @staticmethod
    def _extract_status(results: dict[str, Any], task_id: str) -> str:
        eval_block = results.get("eval", {}) if isinstance(results, dict) else {}
        task_results = eval_block.get(task_id, [])
        if not task_results:
            return "missing"
        status = str(task_results[0].get("status", "missing")).strip().lower()
        return status or "missing"

    @staticmethod
    def _render_eval_observation(status: str, pass_at_k: dict[str, Any], reward_profile: dict[str, Any]) -> str:
        pass_1 = pass_at_k.get("pass@1") if isinstance(pass_at_k, dict) else None
        potential = reward_profile.get("topology_potential")
        value_signal = reward_profile.get("value_signal")
        pass_ratio = reward_profile.get("pass_ratio")
        boundary = reward_profile.get("failure_boundary")
        correction_rule = reward_profile.get("correction_rule")
        return (
            f"[evaluation_status] {status}\n"
            f"[pass@1] {pass_1}\n"
            f"[topology_potential] {potential}\n"
            f"[value_signal] {value_signal}\n"
            f"[pass_ratio] {pass_ratio}\n"
            f"[failure_boundary] {boundary}\n"
            f"[correction_rule] {correction_rule}"
        )

    @staticmethod
    def _evaluator_task_id(task_id: str) -> str:
        return str(task_id).split("/")[-1]

    @staticmethod
    def _local_evaluator_task_id(task_id: str) -> str:
        return str(task_id).strip()
