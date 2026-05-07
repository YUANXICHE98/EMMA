from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from .ablation import AblationSpec, ensure_supported
from .adapter import BenchmarkAdapter, StepResult
from .hypergraph import HypergraphPromptAdapter

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = WORKSPACE_ROOT / "exp"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from modules.encoder import IntentEncoder
from modules.llm_core import FrozenLLM
from modules.memory import EpisodicMemory
from modules.retriever import MemoryRetriever
from modules.rl_optimizer import UniversalRLOptimizer


class ScopedEpisodicMemory(EpisodicMemory):
    def __init__(self, memory_file: Path, config: dict[str, Any] | None = None):
        self.memory_file = str(memory_file)
        self.records = self._load_memory()
        mem_config = (config or {}).get("memory", {}) if isinstance(config, dict) else {}
        self.merge_threshold = 0.99
        self.prune_threshold = float(mem_config.get("prune_threshold", -0.85))
        self.strategy_bonus = float(mem_config.get("strategy_bonus", 0.4))
        print(f"[memory] file={self.memory_file} records={len(self.records)}")


class MemRLBrain:
    supported_ablation_flags = {
        "memory_enabled",
        "failure_memory_enabled",
        "hypergraph_enabled",
        "delta_enabled",
        "forgetting_enabled",
    }
    repairable_failure_prefixes = {
        "axes_",
        "plot_",
        "return_",
        "axis_",
        "numeric_",
        "aggregation_",
        "shape_",
        "wrong_exception_contract",
        "histogram_",
        "column_name_",
    }

    def __init__(self, config: dict[str, Any], results_dir: Path, condition: str):
        self.config = config
        self.results_dir = results_dir
        self.condition = condition

        self.llm = FrozenLLM(config)
        self.primary_model = str(config.get("routing", {}).get("primary_model") or config.get("llm", {}).get("model_name", ""))
        self.secondary_model = str(config.get("routing", {}).get("secondary_model", ""))
        self.routing_enabled = bool(config.get("routing", {}).get("enabled", False) and self.secondary_model)
        self.routing_trigger_mode = str(config.get("routing", {}).get("trigger_mode", "disabled") or "disabled")
        self.secondary_llm = self._build_secondary_llm()
        self.encoder = IntentEncoder(config)
        self.retriever = MemoryRetriever(config)
        self.rl_opt = UniversalRLOptimizer(config)
        self.hypergraph = HypergraphPromptAdapter()
        self.memory_bank = None
        self._seed_initialized = False

    @staticmethod
    def _compact_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return " ".join(text.split())

    @classmethod
    def _structured_embedding_text(cls, structured: dict[str, Any]) -> str:
        if not isinstance(structured, dict):
            return ""
        fields = [
            ("task_type", structured.get("task_type", "")),
            ("memory_level", structured.get("memory_level", "")),
            ("goal", structured.get("goal", "")),
            ("task_domain_family", structured.get("task_domain_family", "")),
            ("input_family", structured.get("input_family", "")),
            ("subject_family", structured.get("subject_family", "")),
            ("action_type", structured.get("action_type", "")),
            ("output_contract_family", structured.get("output_contract_family", "")),
            ("visualization_family", structured.get("visualization_family", "")),
            ("visualization_contract_family", structured.get("visualization_contract_family", "")),
            ("plot_cardinality_rule", structured.get("plot_cardinality_rule", "")),
            ("axes_semantic_rule", structured.get("axes_semantic_rule", "")),
            ("return_container_rule", structured.get("return_container_rule", "")),
            ("return_slot_signature", structured.get("return_slot_signature", "")),
            ("render_pattern_family", structured.get("render_pattern_family", "")),
            ("exception_contract_family", structured.get("exception_contract_family", "")),
            ("failure_boundary", structured.get("failure_boundary", "")),
            ("verifier_check", structured.get("verifier_check", "")),
            ("verifier_failure_summary", structured.get("verifier_failure_summary", "")),
            ("verifier_repair_constraint", structured.get("verifier_repair_constraint", "")),
            ("reasoning_failure_pattern", structured.get("reasoning_failure_pattern", "")),
            ("recompute_operator", structured.get("recompute_operator", "")),
            ("disallowed_shortcut", structured.get("disallowed_shortcut", "")),
            ("next_reasoning_move", structured.get("next_reasoning_move", "")),
            ("proof_obligation", structured.get("proof_obligation", "")),
            ("value_bias", structured.get("value_bias", "")),
            ("abstract_signature", structured.get("abstract_signature", "")),
            ("strategy_key", structured.get("strategy_key", "")),
            ("edge_key", structured.get("edge_key", "")),
        ]
        lines = []
        for key, value in fields:
            compact = cls._compact_text(value)
            if compact:
                lines.append(f"{key}: {compact}")
        return "\n".join(lines)

    @classmethod
    def _task_embedding_text(cls, task: Any, state_text: str, *, state_repr: str = "") -> str:
        metadata = getattr(task, "metadata", {}) or {}
        fields = [
            ("task_type", getattr(task, "task_type", "")),
            ("goal", metadata.get("abstract_goal", "") or getattr(task, "goal_repr", "")),
            ("task_domain_family", metadata.get("task_domain_family", "")),
            ("input_family", metadata.get("input_family", "")),
            ("subject_family", metadata.get("subject_family", "")),
            ("transform_family", metadata.get("transform_family", "")),
            ("output_contract_family", metadata.get("output_contract_family", "")),
            ("visualization_family", metadata.get("visualization_family", "")),
            ("visualization_contract_family", metadata.get("visualization_contract_family", "")),
            ("plot_cardinality_rule", metadata.get("plot_cardinality_rule", "")),
            ("axes_semantic_rule", metadata.get("axes_semantic_rule", "")),
            ("return_container_rule", metadata.get("return_container_rule", "")),
            ("return_slot_signature", metadata.get("return_slot_signature", "")),
            ("render_pattern_family", metadata.get("render_pattern_family", "")),
            ("exception_contract_family", metadata.get("exception_contract_family", "")),
            ("abstract_signature", metadata.get("abstract_signature", "")),
            ("state_repr", state_repr),
            ("state_text", state_text),
        ]
        lines = []
        for key, value in fields:
            compact = cls._compact_text(value)
            if compact:
                lines.append(f"{key}: {compact}")
        return "\n".join(lines)

    def _build_secondary_llm(self) -> FrozenLLM | None:
        if not self.routing_enabled:
            return None
        secondary_config = copy.deepcopy(self.config)
        secondary_config.setdefault("llm", {})
        secondary_config["llm"]["model_name"] = self.secondary_model
        secondary_protocol = str(self.config.get("routing", {}).get("secondary_protocol", "")).strip()
        if secondary_protocol:
            secondary_config["llm"]["protocol"] = secondary_protocol
        return FrozenLLM(secondary_config)

    def activate_ablation(self, ablation_name: str) -> AblationSpec:
        spec = ensure_supported(ablation_name, supported_flags=self.supported_ablation_flags)
        if spec.memory_enabled:
            self.memory_bank = ScopedEpisodicMemory(
                self.results_dir / f"emma_memory_{ablation_name}.json",
                config=self.config,
            )
        return spec

    def initialize_seed_memory(self, adapter: BenchmarkAdapter, ablation: AblationSpec) -> None:
        if self._seed_initialized or self.memory_bank is None or not ablation.memory_enabled:
            return
        seed_specs = adapter.seed_memories()
        if not seed_specs:
            self._seed_initialized = True
            return

        initialized = 0
        for seed in seed_specs:
            if not isinstance(seed, dict):
                continue
            structured = seed.get("structured", {})
            if not isinstance(structured, dict):
                continue
            render_text = seed.get("experience") or self._render_structured_memory(structured)
            state_text = seed.get("state_text") or render_text
            embedding_text = self._structured_embedding_text(structured) or str(state_text)
            z_seed = self.encoder.encode(embedding_text)
            initial_q = float(seed.get("initial_q", 1.0))
            meta = dict(seed.get("meta", {}) or {})
            meta.setdefault("task_type", structured.get("task_type", ""))
            meta.setdefault("success", structured.get("outcome") == "success")
            meta.setdefault("seed_memory", True)
            meta.setdefault("embedding_text", embedding_text)
            self.memory_bank.add_memory(
                z_seed,
                render_text,
                initial_q=initial_q,
                meta=meta,
                structured=structured,
            )
            initialized += 1

        if initialized:
            self.memory_bank.save_memory()
            print(f"[seed] initialized={initialized}")
        self._seed_initialized = True

    def run_episode(
        self,
        adapter: BenchmarkAdapter,
        task_index: int | None,
        max_steps: int,
        ablation: AblationSpec,
        *,
        allow_retrieval: bool = True,
        allow_value_update: bool = True,
        allow_memory_write: bool = True,
    ) -> dict[str, Any]:
        reset = adapter.reset_task(index=task_index)
        task = reset.task
        observation = reset.observation
        state_repr = reset.state_repr or observation
        semantic_task_text = adapter.task_semantic_text(task) or task.instruction
        current_valid_actions = list(reset.candidate_actions or reset.valid_actions)
        anchor_state_text = adapter.memory_state_text(
            task,
            observation,
            current_valid_actions,
            [],
            state_repr=state_repr,
        )
        episode_api_start = self.encoder.get_api_call_count() + self.llm.get_api_call_count()
        anchor_embedding_text = self._task_embedding_text(
            task,
            anchor_state_text or semantic_task_text,
            state_repr=state_repr,
        )
        needs_anchor_embedding = self.memory_bank is not None and (
            allow_retrieval or allow_memory_write or ablation.memory_enabled
        )
        anchor_z = self.encoder.encode(anchor_embedding_text or semantic_task_text) if needs_anchor_embedding else None

        traces: list[dict[str, Any]] = []
        history_log: list[str] = []
        used_memories_timeline: list[dict[str, int]] = []
        reference_only_count = 0
        hypergraph_prompt_injections = 0
        hypergraph_positive_count = 0
        hypergraph_cautionary_count = 0
        routing_escalated = False
        routing_probe: dict[str, Any] = {}
        final = None

        for _ in range(max_steps):
            state_text = adapter.memory_state_text(
                task,
                observation,
                current_valid_actions,
                history_log,
                state_repr=state_repr,
            )
            state_embedding_text = self._task_embedding_text(
                task,
                state_text or semantic_task_text,
                state_repr=state_repr,
            )
            if self.memory_bank is not None and allow_retrieval:
                z_t = self.encoder.encode(state_embedding_text or semantic_task_text)
                m_ctx, used_memory_idx, retrieval_debug = self.retriever.retrieve(
                    z_t,
                    self.memory_bank,
                    task_payload={
                        "task_type": task.task_type,
                        **task.metadata,
                    },
                )
                if used_memory_idx is not None:
                    used_memories_timeline.append({"step": len(traces), "idx": used_memory_idx})
                retrieval_mode = self.retriever._reference_mode_label(m_ctx)
                if retrieval_mode == "reference_only":
                    reference_only_count += 1
            else:
                m_ctx, used_memory_idx, retrieval_debug = None, None, {}
                retrieval_mode = ""

            base_prompt = adapter.build_prompt(task, observation, history_log)
            if getattr(task, "task_type", "") == "code_generation" and traces:
                repair_block = self._code_repair_prompt_block(traces[-1])
                if repair_block:
                    base_prompt = f"{base_prompt}\n\n{repair_block}"
            elif traces:
                repair_block = adapter.build_repair_prompt(task, traces[-1])
                if repair_block:
                    base_prompt = f"{base_prompt}\n\n{repair_block}"
            prompt_memory_context = m_ctx
            if (
                getattr(task, "task_type", "") == "code_generation"
                and traces
                and retrieval_mode == "reference_only"
            ):
                prompt_memory_context = None

            if self.memory_bank is not None and allow_retrieval and ablation.hypergraph_enabled:
                recommendations = self.hypergraph.recommend(
                    memory_bank=self.memory_bank,
                    task=task,
                    history_lines=history_log,
                    valid_actions=current_valid_actions,
                    retrieval_context={
                        "used_memory_idx": used_memory_idx,
                        "retrieval_mode": retrieval_mode,
                    },
                )
                hypergraph_positive_count += len(recommendations.get("positive", []))
                hypergraph_cautionary_count += len(recommendations.get("cautionary", []))
                prompt_block = self.hypergraph.build_prompt_block(recommendations)
                if prompt_block:
                    hypergraph_prompt_injections += 1
                    base_prompt = f"{base_prompt}\n\n{prompt_block}"

            prompt = (
                self.retriever.assemble_prompt(base_prompt, prompt_memory_context, task_type=task.task_type)
                if self.memory_bank is not None
                else base_prompt
            )
            selected_model = self.primary_model or str(self.config.get("llm", {}).get("model_name", ""))
            route_mode = self.routing_trigger_mode
            if self.secondary_llm is not None and route_mode == "difficulty_probe":
                route_hint = adapter.route_hint(task, observation) or {}
                hinted_label = str(route_hint.get("route_label", "")).strip().upper()
                if hinted_label in {"EASY", "HARD"}:
                    route_label = hinted_label
                    route_debug = {
                        "raw_response": hinted_label,
                        "system_prompt": "",
                        "user_prompt": "",
                        "error": "",
                    }
                    trigger_reason = str(route_hint.get("trigger_reason", "")).strip() or (
                        "difficulty_hard" if route_label == "HARD" else "difficulty_easy"
                    )
                else:
                    route_prompt = adapter.route_prompt(task, observation)
                    route_label = self.llm.classify_route(route_prompt)
                    route_debug = self.llm.get_last_route_debug()
                    trigger_reason = "difficulty_hard" if route_label == "HARD" else "difficulty_easy"
                routing_probe = {
                    "should_escalate": route_label == "HARD",
                    "trigger_mode": route_mode,
                    "trigger_reason": trigger_reason,
                    "route_label": route_label,
                    "route_raw": route_debug.get("raw_response", ""),
                    "route_source": route_hint.get("source", "llm_probe") if hinted_label in {"EASY", "HARD"} else "llm_probe",
                }
                if route_label == "HARD":
                    routing_escalated = True
                    raw_action = self.secondary_llm.generate_action(prompt, task_type=task.task_type)
                    llm_debug = self.secondary_llm.get_last_action_debug()
                    selected_model = self.secondary_model
                else:
                    raw_action = self.llm.generate_action(prompt, task_type=task.task_type)
                    llm_debug = self.llm.get_last_action_debug()
            else:
                raw_action = self.llm.generate_action(prompt, task_type=task.task_type)
                llm_debug = self.llm.get_last_action_debug()
                if self.secondary_llm is not None:
                    routing_probe = adapter.route_probe(task, raw_action) or {}
                    if routing_probe.get("should_escalate"):
                        routing_escalated = True
                        raw_action = self.secondary_llm.generate_action(prompt, task_type=task.task_type)
                        llm_debug = self.secondary_llm.get_last_action_debug()
                        selected_model = self.secondary_model
            action = adapter.normalize_action(raw_action)
            llm_infra_label = ""
            if getattr(task, "task_type", "") == "code_generation":
                llm_infra_label = self._llm_infra_failure_label(
                    {
                        "raw_llm_output": llm_debug.get("raw_response", raw_action),
                        "llm_error": llm_debug.get("error", ""),
                    }
                ) or ""
            if getattr(task, "task_type", "") == "code_generation" and not action and llm_infra_label:
                reward_profile = {
                    "reward": 0.0,
                    "topology_potential": 0.0,
                    "value_signal": -1.0,
                    "pass_ratio": 0.0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "total_tests": 0,
                    "failure_boundary": llm_infra_label,
                    "correction_rule": "Infrastructure failure during LLM generation; retry after provider or network recovers.",
                    "raw_feedback_summary": (
                        f"[llm_infra_error] {llm_infra_label}\n"
                        f"[llm_error] {str(llm_debug.get('error', '')).strip()}"
                    )[:4000],
                }
                step_result = StepResult(
                    observation=(
                        f"[evaluation_status] fail\n"
                        f"[pass@1] 0.0\n"
                        f"[topology_potential] 0.0\n"
                        f"[value_signal] -1.0\n"
                        f"[pass_ratio] 0.0\n"
                        f"[failure_boundary] {llm_infra_label}\n"
                        f"[correction_rule] Infrastructure failure during LLM generation; retry after provider or network recovers."
                    ),
                    reward=0.0,
                    done=True,
                    success=False,
                    state_repr="[code_state]\nCurrent episode terminated before benchmark evaluation due to LLM infrastructure failure.",
                    candidate_actions=adapter.get_valid_actions(),
                    failure_signal=llm_infra_label,
                    terminal_status="failure",
                    info={
                        "status": "llm_infra_error",
                        "error": str(llm_debug.get("error", "")).strip(),
                        "raw_feedback_summary": reward_profile["raw_feedback_summary"],
                        "reward_profile": reward_profile,
                    },
                    valid_actions=adapter.get_valid_actions(),
                )
            else:
                step_result = adapter.step(action)
            observation = step_result.observation
            state_repr = step_result.state_repr or observation
            current_valid_actions = list(step_result.candidate_actions or step_result.valid_actions)
            history_log.append(adapter.history_entry(action, observation, step_result))
            traces.append(
                {
                    "goal_repr": task.goal_repr,
                    "state_repr": state_repr,
                    "prompt": prompt,
                    "memory_context": prompt_memory_context,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_debug": retrieval_debug,
                    "raw_llm_output": llm_debug.get("raw_response", raw_action),
                    "llm_error": llm_debug.get("error", ""),
                    "solver_model": selected_model,
                    "routing_probe": routing_probe,
                    "action": action,
                    "obs": observation,
                    "failure_signal": step_result.failure_signal,
                    "terminal_status": step_result.terminal_status,
                    "pddl_reward": step_result.reward,
                    "is_success": step_result.success,
                    "info": step_result.info,
                    "candidate_actions": current_valid_actions,
                    "valid_actions": step_result.valid_actions,
                }
            )
            final = step_result
            continue_after_failure = (
                not step_result.success
                and len(traces) < max_steps
                and (
                    (
                        getattr(task, "task_type", "") == "code_generation"
                        and self._should_attempt_code_repair(step_result.failure_signal)
                    )
                    or bool(adapter.build_repair_prompt(task, traces[-1]))
                )
            )
            if step_result.done and not continue_after_failure:
                break

        if final is None or not final.done:
            final = adapter.force_finish()
            observation = final.observation
            state_repr = final.state_repr or observation
            current_valid_actions = list(final.candidate_actions or final.valid_actions)
            traces.append(
                {
                    "goal_repr": task.goal_repr,
                    "state_repr": state_repr,
                    "action": "submit",
                    "obs": observation,
                    "failure_signal": final.failure_signal,
                    "terminal_status": final.terminal_status,
                    "pddl_reward": final.reward,
                    "is_success": final.success,
                    "info": final.info,
                    "candidate_actions": current_valid_actions,
                    "valid_actions": final.valid_actions,
                }
            )

        final_success = bool(final.success)
        final_trace = traces[-1] if traces else {}
        llm_infra_label = self._llm_infra_failure_label(final_trace)
        infra_failure = str(final.failure_signal or "").strip() == "evaluator_error" or bool(llm_infra_label)
        retrieval_hit_count = len(used_memories_timeline)
        retrieval_reference_count = reference_only_count
        api_calls = (self.encoder.get_api_call_count() + self.llm.get_api_call_count()) - episode_api_start
        if self.secondary_llm is not None:
            api_calls += self.secondary_llm.get_api_call_count()
        cost_efficiency = (float(final.reward) / api_calls) if api_calls > 0 else 0.0
        if traces:
            discounted_returns, _ = self.rl_opt.compute_discounted_returns(traces, final_success)
            if self.memory_bank is not None and allow_value_update and used_memories_timeline and not infra_failure:
                for usage in used_memories_timeline:
                    step_idx = usage["step"]
                    mem_idx = usage["idx"]
                    record_meta = self.memory_bank.records[mem_idx].get("meta", {}) if 0 <= mem_idx < len(self.memory_bank.records) else {}
                    if bool(record_meta.get("seed_memory", False)):
                        continue
                    g_t = discounted_returns[step_idx]
                    q_old = self.memory_bank.records[mem_idx]["q"]
                    q_new, q_feedback = self.rl_opt.update_q(q_old, g_t)
                    self.memory_bank.apply_localized_feedback(
                        mem_idx,
                        q_new,
                        final_success,
                        forgetting_enabled=ablation.forgetting_enabled,
                    )
                    trace_info = traces[step_idx].setdefault("info", {})
                    trace_info["q_update"] = {
                        "q_old": round(float(q_old), 4),
                        "q_new": round(float(q_new), 4),
                        **q_feedback,
                    }

        if (
            self.memory_bank is not None
            and allow_memory_write
            and not infra_failure
            and (final_success or ablation.failure_memory_enabled)
        ):
            structured_memory = self._build_structured_memory(
                task=task,
                ablation=ablation,
                initial_state_repr=reset.state_repr or observation,
                traces=traces,
                final_success=final_success,
                final_failure_signal=final.failure_signal,
                terminal_status=final.terminal_status,
            )
            experience = self._render_structured_memory(structured_memory)
            final_reward_profile = final.info.get("reward_profile", {}) if isinstance(final.info, dict) else {}
            initial_q = self.rl_opt.get_initial_q(
                final_success,
                topology_potential=final_reward_profile.get("topology_potential") if ablation.delta_enabled else None,
            )
            tactical_idx = self.memory_bank.add_memory(
                anchor_z,
                experience,
                initial_q=initial_q,
                structured=structured_memory,
                meta={
                    "task_type": task.task_type,
                    "task_description": task.task_description,
                    "instruction": task.instruction,
                    "success": final_success,
                    "goal_repr": task.goal_repr,
                    "final_failure_signal": final.failure_signal,
                    "terminal_status": final.terminal_status,
                    "action_sequence": [
                        trace["action"]
                        for trace in traces
                        if trace.get("action") and trace.get("action") != "submit"
                    ],
                    "embedding_text": self._structured_embedding_text(structured_memory),
                },
            )
            if final_success:
                strategy_memory = self._build_strategy_memory(structured_memory)
                tactical_meta = self.memory_bank.records[tactical_idx].get("meta", {}) if tactical_idx is not None else {}
                if strategy_memory is not None and int(tactical_meta.get("success_count", 0)) >= 2:
                    strategy_meta = {
                        "task_type": task.task_type,
                        "task_description": task.task_description,
                        "instruction": task.instruction,
                        "success": True,
                        "memory_level": "strategy",
                        "source_tactical_idx": tactical_idx,
                        "embedding_text": self._structured_embedding_text(strategy_memory),
                    }
                    self.memory_bank.consolidate_strategy(anchor_z, strategy_memory, meta=strategy_meta)
            self.memory_bank.save_memory()

        final_reward_profile = final.info.get("reward_profile", {}) if isinstance(final.info, dict) else {}
        if isinstance(final_reward_profile, dict) and llm_infra_label:
            final_reward_profile = dict(final_reward_profile)
            final_reward_profile["reward"] = 0.0
            final_reward_profile["topology_potential"] = 0.0
            final_reward_profile["value_signal"] = -1.0
            final_reward_profile["passed_tests"] = 0
            final_reward_profile["failed_tests"] = 0
            final_reward_profile["total_tests"] = 0
            final_reward_profile["failure_boundary"] = llm_infra_label
            final_reward_profile["correction_rule"] = "Infrastructure failure during LLM generation; retry after provider or network recovers."
            raw_feedback_summary = str(final_reward_profile.get("raw_feedback_summary", "") or "").strip()
            if raw_feedback_summary:
                final_reward_profile["raw_feedback_summary"] = (
                    f"[llm_infra_error] {llm_infra_label}\n[llm_error] {str(final_trace.get('llm_error', '')).strip()}\n\n"
                    f"{raw_feedback_summary}"
                )[:4000]
            else:
                final_reward_profile["raw_feedback_summary"] = (
                    f"[llm_infra_error] {llm_infra_label}\n[llm_error] {str(final_trace.get('llm_error', '')).strip()}"
                )[:4000]
        result_failure_signal = llm_infra_label or final.failure_signal
        final_reward_value = final_reward_profile.get("reward", final.reward) if isinstance(final_reward_profile, dict) else final.reward
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "task_description": task.task_description,
            "instruction": task.instruction,
            "goal_repr": task.goal_repr,
            "metadata": task.metadata,
            "reward": round(float(final_reward_value), 4),
            "success": final_success,
            "steps": len(traces),
            "final_state_repr": state_repr,
            "final_failure_signal": result_failure_signal,
            "failure_boundary": str(final_reward_profile.get("failure_boundary", "")).strip(),
            "predicted_boundary": str(final_reward_profile.get("predicted_boundary", "")).strip(),
            "predicted_boundary_severity": final_reward_profile.get("predicted_boundary_severity"),
            "observed_boundary": str(final_reward_profile.get("observed_boundary", "")).strip(),
            "observed_boundary_severity": final_reward_profile.get("observed_boundary_severity"),
            "correction_rule": str(final_reward_profile.get("correction_rule", "")).strip(),
            "disallowed_shortcut": str(final_reward_profile.get("disallowed_shortcut", "")).strip(),
            "topology_potential": final_reward_profile.get("topology_potential"),
            "value_signal": final_reward_profile.get("value_signal"),
            "final_reward_profile": final_reward_profile,
            "terminal_status": final.terminal_status,
            "retrieval_hit_count": retrieval_hit_count,
            "retrieval_hit_ratio": round(retrieval_hit_count / len(traces), 4) if traces else 0.0,
            "retrieval_reference_count": retrieval_reference_count,
            "retrieval_reference_ratio": round(retrieval_reference_count / len(traces), 4) if traces else 0.0,
            "hypergraph_prompt_injections": hypergraph_prompt_injections,
            "hypergraph_positive_count": hypergraph_positive_count,
            "hypergraph_cautionary_count": hypergraph_cautionary_count,
            "routing_enabled": self.routing_enabled,
            "routing_trigger_mode": self.routing_trigger_mode,
            "routing_escalated": routing_escalated,
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "api_calls": api_calls,
            "cost_efficiency": round(cost_efficiency, 4),
            "trajectory": traces,
        }

    @staticmethod
    def _code_repair_prompt_block(last_trace: dict[str, Any]) -> str:
        if not isinstance(last_trace, dict):
            return ""
        failure_signal = str(last_trace.get("failure_signal", "")).strip()
        if not failure_signal or failure_signal == "evaluator_error":
            return ""
        info = last_trace.get("info", {}) if isinstance(last_trace.get("info"), dict) else {}
        reward_profile = info.get("reward_profile", {}) if isinstance(info.get("reward_profile"), dict) else {}
        correction_rule = str(reward_profile.get("correction_rule", "")).strip()
        last_code = str(last_trace.get("action", "")).strip()
        if not last_code:
            return ""
        literal_contract_note = ""
        if "plot_ylabel_contract_mismatch" in {part for part in failure_signal.split("+") if part}:
            literal_contract_note = (
                "[literal_contract_priority]\n"
                "This repair is a literal string contract, not a semantic approximation.\n"
                "Keep the plotting structure unchanged unless needed, and change the y-axis label to the exact evaluator-required string.\n"
                "Do not use synonyms, near-matches, or stylistic rewrites for the label text.\n\n"
            )
        return (
            "[EMMA Local Repair Mode]\n"
            "You are repairing the most recent code attempt, not restarting from scratch.\n"
            "Keep the previous implementation structure unless the verifier feedback proves that a specific local part is wrong.\n"
            "Apply the smallest change set that repairs the current failure boundary.\n"
            "Do not introduce new plotting APIs, new control kwargs, or new output reshaping unless the correction rule explicitly requires it.\n"
            "Preserve already-correct parts such as imports, function signature, dataframe preprocessing, and working transform steps.\n"
            f"[repair_target_failure_boundary]\n{failure_signal}\n\n"
            f"[repair_rule]\n{correction_rule}\n\n"
            f"{literal_contract_note}"
            "[previous_submission]\n"
            f"{last_code}"
        )

    @classmethod
    def _should_attempt_code_repair(cls, failure_signal: str) -> bool:
        signal = str(failure_signal or "").strip()
        if not signal or signal == "evaluator_error":
            return False
        if signal == "test_failure":
            return False
        if signal == "lexical_ban_violation":
            return True
        return any(signal.startswith(prefix) for prefix in cls.repairable_failure_prefixes)

    @staticmethod
    def _normalize_action_type(task: Any, traces: list[dict[str, Any]]) -> str:
        if getattr(task, "task_type", "") == "code_generation":
            return "write_self_contained_solution"
        first_trace = traces[0] if traces else {}
        candidate_actions = first_trace.get("candidate_actions") or []
        if candidate_actions:
            return str(candidate_actions[0])
        action = str(first_trace.get("action", "")).strip()
        if not action:
            return "unknown_action"
        first_line = action.splitlines()[0].strip()
        return first_line[:120] or "unknown_action"

    @classmethod
    def _build_structured_memory(
        cls,
        *,
        task: Any,
        ablation: AblationSpec,
        initial_state_repr: str,
        traces: list[dict[str, Any]],
        final_success: bool,
        final_failure_signal: str,
        terminal_status: str,
    ) -> dict[str, Any]:
        first_trace = traces[0] if traces else {}
        final_trace = traces[-1] if traces else {}
        outcome_label = "success" if final_success else "failure"
        raw_failure_boundary = final_failure_signal or str(final_trace.get("failure_signal", "")).strip() or "none"
        reward_profile = final_trace.get("info", {}).get("reward_profile", {}) if isinstance(final_trace.get("info"), dict) else {}
        verifier_evidence = cls._collect_verifier_evidence(traces)
        failure_boundary = raw_failure_boundary
        correction_rule = str(reward_profile.get("correction_rule", "")).strip()
        topology_potential = reward_profile.get("topology_potential")
        value_signal = reward_profile.get("value_signal")
        success_contract_rule = cls._success_contract_rule(task)
        structured_memory = {
            "memory_schema": "emma_structured_v1",
            "memory_level": "tactical",
            "task_type": getattr(task, "task_type", "") or "unknown_task",
            "goal": str(getattr(task, "metadata", {}).get("abstract_goal", "") or getattr(task, "goal_repr", "")).strip(),
            "precondition_or_state": initial_state_repr,
            "action_type": str(getattr(task, "metadata", {}).get("transform_family", "") or cls._normalize_action_type(task, traces)).strip(),
            "outcome": outcome_label,
            "terminal_status": terminal_status or str(final_trace.get("terminal_status", "")).strip() or outcome_label,
            "failure_boundary": failure_boundary,
            "verifier_check": verifier_evidence.get("check", ""),
            "verifier_valid": verifier_evidence.get("valid", ""),
            "verifier_failure_summary": verifier_evidence.get("summary", ""),
            "verifier_repair_constraint": verifier_evidence.get("repair_constraint", ""),
            "correction_rule": correction_rule,
            "success_contract_rule": success_contract_rule,
            "reasoning_failure_pattern": str(reward_profile.get("reasoning_failure_pattern", "")).strip(),
            "recompute_operator": str(reward_profile.get("recompute_operator", "")).strip(),
            "disallowed_shortcut": str(reward_profile.get("disallowed_shortcut", "")).strip(),
            "next_reasoning_move": str(reward_profile.get("next_reasoning_move", "")).strip(),
            "proof_obligation": str(reward_profile.get("proof_obligation", "")).strip(),
            "value_bias": "positive_reuse" if final_success else "cautionary_avoid",
            "input_family": str(getattr(task, "metadata", {}).get("input_family", "")).strip(),
            "subject_family": str(getattr(task, "metadata", {}).get("subject_family", "")).strip(),
            "column_preservation_rule": str(getattr(task, "metadata", {}).get("column_preservation_rule", "")).strip(),
            "aggregation_rule": str(getattr(task, "metadata", {}).get("aggregation_rule", "")).strip(),
            "output_contract_family": str(getattr(task, "metadata", {}).get("output_contract_family", "")).strip(),
            "visualization_family": str(getattr(task, "metadata", {}).get("visualization_family", "")).strip(),
            "visualization_contract_family": str(
                getattr(task, "metadata", {}).get("visualization_contract_family", "")
            ).strip(),
            "plot_cardinality_rule": str(getattr(task, "metadata", {}).get("plot_cardinality_rule", "")).strip(),
            "axes_semantic_rule": str(getattr(task, "metadata", {}).get("axes_semantic_rule", "")).strip(),
            "return_container_rule": str(getattr(task, "metadata", {}).get("return_container_rule", "")).strip(),
            "return_slot_signature": str(getattr(task, "metadata", {}).get("return_slot_signature", "")).strip(),
            "forbidden_patterns": str(getattr(task, "metadata", {}).get("forbidden_patterns", "")).strip(),
            "lexical_bans": str(getattr(task, "metadata", {}).get("lexical_bans", "")).strip(),
            "render_pattern_family": str(getattr(task, "metadata", {}).get("render_pattern_family", "")).strip(),
            "render_pattern_hint": str(getattr(task, "metadata", {}).get("render_pattern_hint", "")).strip(),
            "exception_contract_family": str(
                getattr(task, "metadata", {}).get("exception_contract_family", "")
            ).strip(),
            "abstract_signature": str(getattr(task, "metadata", {}).get("abstract_signature", "")).strip(),
            "edge_key": (
                f"{getattr(task, 'metadata', {}).get('abstract_signature', '')}"
                f"|{failure_boundary}|{verifier_evidence.get('check', '')}|{outcome_label}"
            ),
            "strategy_key": (
                f"{getattr(task, 'metadata', {}).get('transform_family', '')}"
                f"|{getattr(task, 'metadata', {}).get('output_contract_family', '')}"
                f"|{getattr(task, 'metadata', {}).get('visualization_family', '')}"
                f"|{verifier_evidence.get('check', '')}"
            ),
            "evidence": {
                "steps": len(traces),
                "first_action_family": str(getattr(task, "metadata", {}).get("transform_family", "")).strip()
                or cls._normalize_action_type(task, traces),
                "last_observation": str(final_trace.get("obs", "")).strip()[:160],
                "topology_potential": topology_potential,
                "value_signal": value_signal,
                "pass_ratio": reward_profile.get("pass_ratio"),
                "verifier_evidence": verifier_evidence,
            },
        }
        return structured_memory

    @staticmethod
    def _collect_verifier_evidence(traces: list[dict[str, Any]]) -> dict[str, Any]:
        collected: list[dict[str, Any]] = []
        for trace in traces or []:
            info = trace.get("info", {}) if isinstance(trace, dict) else {}
            if not isinstance(info, dict):
                continue
            feedback = info.get("verifiable_feedback", {})
            if not isinstance(feedback, dict) or not feedback:
                continue
            collected.append(feedback)
        if not collected:
            return {}
        feedback = collected[-1]
        summary_parts = []
        for key in (
            "evidence_summary",
            "claim_error",
            "sample_substitution",
            "parameter_values",
            "claimed_verification",
        ):
            value = str(feedback.get(key, "")).strip()
            if value:
                summary_parts.append(f"{key}: {value}")
        if not summary_parts:
            check = str(feedback.get("check", "")).strip()
            candidate = str(feedback.get("candidate", "")).strip()
            remainder = str(feedback.get("remainder", "")).strip()
            if check:
                summary_parts.append(f"check: {check}")
            if candidate:
                summary_parts.append(f"candidate: {candidate}")
            if remainder:
                summary_parts.append(f"remainder: {remainder}")
        if len(collected) > 1:
            failed_candidates = []
            seen_candidates: set[str] = set()
            for evidence in collected:
                if evidence.get("valid") is not False:
                    continue
                candidate = str(evidence.get("candidate", "")).strip()
                check = str(evidence.get("check", "")).strip()
                remainder = str(evidence.get("remainder", "")).strip()
                if not candidate or candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                if remainder:
                    failed_candidates.append(f"{candidate} failed {check} with remainder {remainder}")
                elif check:
                    failed_candidates.append(f"{candidate} failed {check}")
                else:
                    failed_candidates.append(candidate)
            if failed_candidates:
                summary_parts.append("all_failed_candidates: " + "; ".join(failed_candidates))
        return {
            "check": str(feedback.get("check", "")).strip(),
            "valid": feedback.get("valid", ""),
            "summary": "; ".join(summary_parts)[:800],
            "repair_constraint": str(feedback.get("repair_constraint", "")).strip()[:800],
            "raw": feedback,
        }

    @staticmethod
    def _llm_infra_failure_label(final_trace: dict[str, Any]) -> str:
        llm_error = str(final_trace.get("llm_error", "") or "").strip().lower()
        raw_llm_output = str(final_trace.get("raw_llm_output", "") or "").strip()
        if raw_llm_output or not llm_error:
            return ""
        if "gateway time-out" in llm_error or "504" in llm_error:
            return "llm_gateway_timeout"
        if "timed out" in llm_error or "timeout" in llm_error:
            return "llm_timeout"
        if "connection error" in llm_error:
            return "llm_connection_error"
        return "llm_infra_error"

    @classmethod
    def _build_strategy_memory(cls, tactical_memory: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(tactical_memory, dict):
            return None
        if tactical_memory.get("outcome") != "success":
            return None
        strategy_key = str(tactical_memory.get("strategy_key", "")).strip()
        if not strategy_key:
            return None
        strategy_memory = {
            "memory_schema": "emma_strategy_v1",
            "memory_level": "strategy",
            "task_type": tactical_memory.get("task_type", ""),
            "goal": tactical_memory.get("goal", ""),
            "precondition_or_state": (
                f"{tactical_memory.get('input_family', '')}"
                f" -> {tactical_memory.get('output_contract_family', '')}"
            ),
            "action_type": tactical_memory.get("action_type", ""),
            "outcome": "success",
            "terminal_status": "success",
            "failure_boundary": "reuse_only_when_contract_matches",
            "correction_rule": "",
            "value_bias": "positive_reuse",
            "trusted": True,
            "input_family": tactical_memory.get("input_family", ""),
            "column_preservation_rule": tactical_memory.get("column_preservation_rule", ""),
            "aggregation_rule": tactical_memory.get("aggregation_rule", ""),
            "output_contract_family": tactical_memory.get("output_contract_family", ""),
            "visualization_family": tactical_memory.get("visualization_family", ""),
            "visualization_contract_family": tactical_memory.get("visualization_contract_family", ""),
            "plot_cardinality_rule": tactical_memory.get("plot_cardinality_rule", ""),
            "axes_semantic_rule": tactical_memory.get("axes_semantic_rule", ""),
            "return_container_rule": tactical_memory.get("return_container_rule", ""),
            "return_slot_signature": tactical_memory.get("return_slot_signature", ""),
            "forbidden_patterns": tactical_memory.get("forbidden_patterns", ""),
            "lexical_bans": tactical_memory.get("lexical_bans", ""),
            "render_pattern_family": tactical_memory.get("render_pattern_family", ""),
            "render_pattern_hint": tactical_memory.get("render_pattern_hint", ""),
            "exception_contract_family": tactical_memory.get("exception_contract_family", ""),
            "success_contract_rule": tactical_memory.get("success_contract_rule", ""),
            "abstract_signature": strategy_key,
            "edge_key": f"strategy::{strategy_key}",
            "strategy_key": strategy_key,
            "consolidation_count": 1,
            "parent_edge_keys": [tactical_memory.get("edge_key", "")],
            "evidence": {
                "steps": tactical_memory.get("evidence", {}).get("steps", 0),
                "first_action_family": tactical_memory.get("evidence", {}).get("first_action_family", ""),
                "last_observation": tactical_memory.get("evidence", {}).get("last_observation", ""),
            },
        }
        strategy_memory["rendered_hint"] = cls._render_structured_memory(strategy_memory)
        strategy_memory["q_init"] = 1.0
        return strategy_memory

    @staticmethod
    def _render_structured_memory(structured_memory: dict[str, Any]) -> str:
        evidence = structured_memory.get("evidence", {}) or {}
        return (
            "[EMMA Structured Memory]\n"
            f"[schema]\n{structured_memory.get('memory_schema', 'emma_structured_v1')}\n\n"
            f"[task_type]\n{structured_memory.get('task_type', '')}\n\n"
            f"[goal]\n{structured_memory.get('goal', '')}\n\n"
            f"[precondition_or_state]\n{structured_memory.get('precondition_or_state', '')}\n\n"
            f"[action_type]\n{structured_memory.get('action_type', '')}\n\n"
            f"[input_family]\n{structured_memory.get('input_family', '')}\n\n"
            f"[subject_family]\n{structured_memory.get('subject_family', '')}\n\n"
            f"[disallowed_shortcut]\n{structured_memory.get('disallowed_shortcut', '')}\n\n"
            f"[column_preservation_rule]\n{structured_memory.get('column_preservation_rule', '')}\n\n"
            f"[aggregation_rule]\n{structured_memory.get('aggregation_rule', '')}\n\n"
            f"[output_contract_family]\n{structured_memory.get('output_contract_family', '')}\n\n"
            f"[visualization_family]\n{structured_memory.get('visualization_family', '')}\n\n"
            f"[visualization_contract_family]\n{structured_memory.get('visualization_contract_family', '')}\n\n"
            f"[plot_cardinality_rule]\n{structured_memory.get('plot_cardinality_rule', '')}\n\n"
            f"[axes_semantic_rule]\n{structured_memory.get('axes_semantic_rule', '')}\n\n"
            f"[return_container_rule]\n{structured_memory.get('return_container_rule', '')}\n\n"
            f"[return_slot_signature]\n{structured_memory.get('return_slot_signature', '')}\n\n"
            f"[forbidden_patterns]\n{structured_memory.get('forbidden_patterns', '')}\n\n"
            f"[lexical_bans]\n{structured_memory.get('lexical_bans', '')}\n\n"
            f"[render_pattern_family]\n{structured_memory.get('render_pattern_family', '')}\n\n"
            f"[render_pattern_hint]\n{structured_memory.get('render_pattern_hint', '')}\n\n"
            f"[exception_contract_family]\n{structured_memory.get('exception_contract_family', '')}\n\n"
            f"[abstract_signature]\n{structured_memory.get('abstract_signature', '')}\n\n"
            f"[outcome]\n{structured_memory.get('outcome', '')}\n\n"
            f"[terminal_status]\n{structured_memory.get('terminal_status', '')}\n\n"
            f"[failure_boundary]\n{structured_memory.get('failure_boundary', '')}\n\n"
            f"[verifier_check]\n{structured_memory.get('verifier_check', '')}\n\n"
            f"[verifier_valid]\n{structured_memory.get('verifier_valid', '')}\n\n"
            f"[verifier_failure_summary]\n{structured_memory.get('verifier_failure_summary', '')}\n\n"
            f"[verifier_repair_constraint]\n{structured_memory.get('verifier_repair_constraint', '')}\n\n"
            f"[reasoning_failure_pattern]\n{structured_memory.get('reasoning_failure_pattern', '')}\n\n"
            f"[recompute_operator]\n{structured_memory.get('recompute_operator', '')}\n\n"
            f"[disallowed_shortcut]\n{structured_memory.get('disallowed_shortcut', '')}\n\n"
            f"[next_reasoning_move]\n{structured_memory.get('next_reasoning_move', '')}\n\n"
            f"[proof_obligation]\n{structured_memory.get('proof_obligation', '')}\n\n"
            f"[correction_rule]\n{structured_memory.get('correction_rule', '')}\n\n"
            f"[success_contract_rule]\n{structured_memory.get('success_contract_rule', '')}\n\n"
            f"[value_bias]\n{structured_memory.get('value_bias', '')}\n\n"
            f"[evidence_steps]\n{evidence.get('steps', 0)}\n\n"
            f"[evidence_first_action_family]\n{evidence.get('first_action_family', '')}\n\n"
            f"[evidence_topology_potential]\n{evidence.get('topology_potential', '')}\n\n"
            f"[evidence_value_signal]\n{evidence.get('value_signal', '')}\n\n"
            f"[evidence_pass_ratio]\n{evidence.get('pass_ratio', '')}\n\n"
            f"[evidence_last_observation]\n{evidence.get('last_observation', '')}"
        )

    @staticmethod
    def _success_contract_rule(task: Any) -> str:
        metadata = getattr(task, "metadata", {}) or {}
        if getattr(task, "task_type", "") != "code_generation":
            if getattr(task, "task_type", "") != "closed_ended_reasoning":
                return ""
            answer_contract = str(metadata.get("answer_contract_family", "")).strip()
            question_form = str(metadata.get("question_form_family", "")).strip()
            reasoning_family = str(metadata.get("reasoning_family", "")).strip()
            rules: list[str] = []
            if answer_contract == "numeric_exact_contract":
                rules.append("emit one fully evaluated numeric Exact Answer only after an internal derivation yields that exact value")
                if question_form == "numeric_closed_form_query":
                    rules.append("derive or identify the governing formula for the exact object class before substituting parameters")
                if "symbolic_or_numeric_derivation" in reasoning_family:
                    rules.append("keep an internal audit chain theorem or invariant -> instantiated expression -> evaluated number")
            elif answer_contract == "symbolic_exact_contract":
                rules.append("emit one compact symbolic Exact Answer line only after checking branch or family completeness")
            elif answer_contract == "multiple_choice_contract":
                rules.append("emit exactly one option letter or exact option text after eliminating incompatible choices")
            elif answer_contract == "text_exact_contract":
                rules.append("emit one terse entity/fact answer selected by the decisive cue in the prompt")
            if not rules:
                return ""
            return "Success reuse contract: " + "; ".join(rules) + "."

        input_family = str(metadata.get("input_family", "")).strip()
        column_preservation_rule = str(metadata.get("column_preservation_rule", "")).strip()
        matrix_operation_rule = str(metadata.get("matrix_operation_rule", "")).strip()
        aggregation_rule = str(metadata.get("aggregation_rule", "")).strip()
        transform_family = str(metadata.get("transform_family", "")).strip()
        output_contract_family = str(metadata.get("output_contract_family", "")).strip()
        render_pattern_family = str(metadata.get("render_pattern_family", "")).strip()
        visualization_family = str(metadata.get("visualization_family", "")).strip()
        plot_cardinality_rule = str(metadata.get("plot_cardinality_rule", "")).strip()
        axes_semantic_rule = str(metadata.get("axes_semantic_rule", "")).strip()
        return_container_rule = str(metadata.get("return_container_rule", "")).strip()
        return_slot_signature = str(metadata.get("return_slot_signature", "")).strip()
        forbidden_patterns = str(metadata.get("forbidden_patterns", "")).strip()
        lexical_bans = str(metadata.get("lexical_bans", "")).strip()

        rules: list[str] = []
        if input_family == "dataframe_input":
            rules.append("operate on the numeric dataframe slice rather than on the raw mixed dataframe whenever a numeric transform or plot is required")
        elif input_family == "numeric_matrix_input":
            rules.append("treat the input as a numeric matrix / ndarray first, and only convert to DataFrame after the required matrix-level computation is complete")
        if column_preservation_rule:
            rules.append(column_preservation_rule)
        if matrix_operation_rule:
            rules.append(matrix_operation_rule)
        if aggregation_rule:
            rules.append(aggregation_rule)
        if transform_family == "columnwise_standardization":
            rules.append("repair missing values before computing z-scores and compute z-scores columnwise on the preserved dataframe column set")
        elif transform_family == "feature_standardization":
            rules.append("standardize the numeric feature matrix and preserve column alignment in the returned dataframe")
        elif transform_family == "grouped_unique_value_count":
            rules.append("group by the non-target key columns and compute the unique-value count of the target value column before plotting")
        elif transform_family == "rowwise_distribution_statistic":
            rules.append("compute the rowwise statistic directly on the matrix input with an axis-level operation before wrapping the result in a DataFrame")

        if render_pattern_family == "dataframe_native_hist_collection":
            rules.append("use one dataframe-native histogram grid call on the transformed numeric dataframe")
            if "len(plots[0])" in plot_cardinality_rule or "plots[0]" in axes_semantic_rule:
                rules.append("prefer DataFrame.hist with an explicit single-row layout over DataFrame.plot.hist(subplots=True) for row-grid histogram contracts")
                rules.append("preserve the evaluator-facing row-grid axes container returned by the histogram call instead of flattening it into a plain list")
                rules.append("for row-grid histogram contracts, keep a single-row layout whose first row enumerates the numeric-column histogram axes")
            else:
                rules.append("normalize the histogram renderer output after the plotting call into the required flat axes collection")
                rules.append("when the renderer returns an array-like grid, flatten it after the call and keep only the axes that correspond to numeric columns")
            rules.append("do not request output normalization through invented return-control kwargs such as return_axes")
        elif render_pattern_family == "single_axes_hist_plot":
            rules.append("use a single histogram plotting call that returns one Axes object directly")
            rules.append("preserve the native single-Axes return form instead of flattening, indexing, slicing, or wrapping it into a collection")
        elif render_pattern_family == "hist_container_return":
            rules.append("use the histogram API whose native return matches the evaluator-expected plot container")
            rules.append("preserve the native histogram return container instead of collapsing it into one Axes object")
        elif render_pattern_family == "dataframe_native_plot_accessor":
            rules.append("prefer the dataframe-native plotting accessor when it already satisfies the output contract")
            rules.append("return the native plotting object in the exact structure expected by the benchmark")

        if output_contract_family == "dataframe_plus_axes_collection":
            rules.append("return a tuple of transformed dataframe and axes collection with no extra conversion layer")
        elif output_contract_family == "dataframe_plus_single_axes":
            rules.append("return a tuple of transformed dataframe and one Axes object with no extra conversion layer")
        elif output_contract_family == "dataframe_plus_plot_container":
            rules.append("return a tuple of transformed dataframe and the native plot container with no extra conversion layer")
        if plot_cardinality_rule:
            rules.append(plot_cardinality_rule)
        if axes_semantic_rule:
            rules.append(axes_semantic_rule)
        if return_container_rule:
            rules.append(return_container_rule)
        if return_slot_signature:
            rules.append(f"preserve return slot order exactly as {return_slot_signature}")
        if forbidden_patterns:
            rules.append(forbidden_patterns)
        if lexical_bans:
            rules.append(f"hard lexical bans: {lexical_bans}")
        if visualization_family == "heatmap":
            rules.append("keep the plotting path aligned with a heatmap renderer instead of switching to histogram or line APIs")
        elif visualization_family == "histogram":
            rules.append("keep the plotting path aligned with histogram rendering instead of switching to unrelated plot families")

        if not rules:
            return ""
        return "Success reuse contract: " + "; ".join(rules) + "."

    @staticmethod
    def _fallback_experience_summary(
        task_instruction: str,
        traces: list[dict[str, Any]],
        final_success: bool,
        error: Exception,
    ) -> str:
        recent_steps = []
        for idx, trace in enumerate(traces[-5:], start=max(len(traces) - 4, 1)):
            action = trace.get("action", "")
            obs = str(trace.get("obs", ""))[:120]
            recent_steps.append(f"{idx}. action={action} | obs={obs}")
        status = "success" if final_success else "failure"
        recent_block = "\n".join(recent_steps) if recent_steps else "None"
        return (
            "Meta State-Transition Rule: fallback summary because LLM experience summarization was unavailable.\n"
            f"Task: {task_instruction}\n"
            f"Outcome: {status}\n"
            f"Fallback reason: {error}\n"
            "Recent trajectory:\n"
            f"{recent_block}"
        )

    def save_summary(
        self,
        benchmark_name: str,
        ablation: AblationSpec,
        start_index: int,
        results: list[dict[str, Any]],
        results_file: Path,
    ) -> None:
        mean_reward = sum(item["reward"] for item in results) / len(results) if results else 0.0
        exact_successes = sum(1 for item in results if item["success"])
        mean_cost_efficiency = sum(item.get("cost_efficiency", 0.0) for item in results) / len(results) if results else 0.0
        mean_retrieval_hit_ratio = sum(item.get("retrieval_hit_ratio", 0.0) for item in results) / len(results) if results else 0.0
        mean_retrieval_reference_ratio = (
            sum(item.get("retrieval_reference_ratio", 0.0) for item in results) / len(results) if results else 0.0
        )
        mean_hypergraph_prompt_injections = (
            sum(item.get("hypergraph_prompt_injections", 0.0) for item in results) / len(results) if results else 0.0
        )
        mean_hypergraph_positive_count = (
            sum(item.get("hypergraph_positive_count", 0.0) for item in results) / len(results) if results else 0.0
        )
        mean_hypergraph_cautionary_count = (
            sum(item.get("hypergraph_cautionary_count", 0.0) for item in results) / len(results) if results else 0.0
        )
        final_failure_signal_counts: dict[str, int] = {}
        observed_boundary_counts: dict[str, int] = {}
        predicted_boundary_counts: dict[str, int] = {}
        protocol_delta_blocked_count = 0
        for item in results:
            final_failure_signal = str(item.get("final_failure_signal", "")).strip() or "none"
            final_failure_signal_counts[final_failure_signal] = final_failure_signal_counts.get(final_failure_signal, 0) + 1
            observed_boundary = str(item.get("observed_boundary", "")).strip()
            if observed_boundary:
                observed_boundary_counts[observed_boundary] = observed_boundary_counts.get(observed_boundary, 0) + 1
            predicted_boundary = str(item.get("predicted_boundary", "")).strip()
            if predicted_boundary:
                predicted_boundary_counts[predicted_boundary] = predicted_boundary_counts.get(predicted_boundary, 0) + 1
            if item.get("protocol_delta_blocked"):
                protocol_delta_blocked_count += 1
        judge_mode = str(self.config.get("runner", {}).get("judge_mode", ""))
        judge_model = str(self.config.get("runner", {}).get("judge_model", "")) if judge_mode == "llm_judge" else ""
        routing_escalation_count = sum(1 for item in results if item.get("routing_escalated"))
        summary = {
            "benchmark": benchmark_name,
            "condition": ablation.name,
            "episodes": len(results),
            "start_index": start_index,
            "solver_model": str(self.config.get("llm", {}).get("model_name", "")),
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "routing_enabled": self.routing_enabled,
            "routing_trigger_mode": self.routing_trigger_mode,
            "routing_escalation_count": routing_escalation_count,
            "judge_mode": judge_mode,
            "judge_model": judge_model,
            "embedding_model": str(self.config.get("encoder", {}).get("model_name", "")),
            "mean_reward": round(mean_reward, 4),
            "exact_success_rate": round(exact_successes / len(results), 4) if results else 0.0,
            "mean_cost_efficiency": round(mean_cost_efficiency, 4),
            "mean_retrieval_hit_ratio": round(mean_retrieval_hit_ratio, 4),
            "mean_retrieval_reference_ratio": round(mean_retrieval_reference_ratio, 4),
            "mean_hypergraph_prompt_injections": round(mean_hypergraph_prompt_injections, 4),
            "mean_hypergraph_positive_count": round(mean_hypergraph_positive_count, 4),
            "mean_hypergraph_cautionary_count": round(mean_hypergraph_cautionary_count, 4),
            "final_failure_signal_counts": final_failure_signal_counts,
            "observed_boundary_counts": observed_boundary_counts,
            "predicted_boundary_counts": predicted_boundary_counts,
            "protocol_delta_blocked_count": protocol_delta_blocked_count,
            "brain_backend": "shared_emma_brain",
            "memory_scope": "benchmark_private",
            "memory_file": str(self.results_dir / f"emma_memory_{ablation.name}.json"),
            "supported_mechanisms_now": sorted(self.supported_ablation_flags),
            "results": results,
        }
        results_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
