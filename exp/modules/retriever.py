from __future__ import annotations

import numpy as np
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

class MemoryRetriever:
    def __init__(self, config):
        self.config = config
        mem_config = config.get('memory', {})

        # 🔥 核心双阈值引擎 🔥
        self.retrieval_threshold = mem_config.get('retrieval_threshold', 0.8)
        self.merge_threshold = mem_config.get('merge_threshold', 0.99)
        self.semantic_weight = float(mem_config.get('semantic_weight', 0.45))
        self.structure_weight = float(mem_config.get('structure_weight', 0.35))
        self.q_weight = float(mem_config.get('q_weight', 0.20))
        self.q_margin_threshold = float(mem_config.get('q_margin_threshold', 0.18))
        self.combined_margin_threshold = float(mem_config.get('combined_margin_threshold', 0.08))

        print(colored(f"防误导阈值: {self.retrieval_threshold} | 合并阈值: {self.merge_threshold}", "cyan", attrs=['bold']))

    @staticmethod
    def _normalized_q(record):
        if not isinstance(record, dict):
            return 0.0
        q = float(record.get('q', 0.0) or 0.0)
        return max(0.0, min(1.0, (q + 1.0) / 2.0))

    @staticmethod
    def _is_code_near_boundary_task(task_payload):
        if not isinstance(task_payload, dict):
            return False
        if str(task_payload.get("task_type", "")).strip() != "code_generation":
            return False
        visual_contract = str(task_payload.get("visualization_contract_family", "")).strip()
        has_plot_text_contract = any(
            str(task_payload.get(key, "")).strip()
            for key in ("plot_title_rule", "evaluator_contract_note")
        )
        return visual_contract == "explicit_plot_labels_and_titles" or has_plot_text_contract

    @staticmethod
    def _code_contract_compatible(task_payload, structured):
        if not isinstance(task_payload, dict) or not isinstance(structured, dict):
            return True
        if str(task_payload.get("task_type", "")).strip() != "code_generation":
            return True

        task_domain = str(task_payload.get("task_domain_family", "")).strip()
        memory_domain = str(structured.get("task_domain_family", "")).strip()
        if task_domain and memory_domain and task_domain != memory_domain:
            return False

        keys = [
            "output_contract_family",
            "visualization_family",
            "visualization_contract_family",
            "render_pattern_family",
            "exception_contract_family",
            "return_slot_signature",
        ]
        for key in keys:
            lhs = str(task_payload.get(key, "")).strip()
            rhs = str(structured.get(key, "")).strip()
            if lhs and rhs and lhs != rhs:
                return False

        lhs_transform = str(task_payload.get("transform_family", "")).strip()
        rhs_transform = str(structured.get("action_type", "")).strip()
        if lhs_transform and rhs_transform and lhs_transform != rhs_transform:
            return False
        return True

    @staticmethod
    def _code_reference_compatible(task_payload, structured):
        if not isinstance(task_payload, dict) or not isinstance(structured, dict):
            return True
        if str(task_payload.get("task_type", "")).strip() != "code_generation":
            return True

        task_domain = str(task_payload.get("task_domain_family", "")).strip()
        memory_domain = str(structured.get("task_domain_family", "")).strip()
        if task_domain and memory_domain and task_domain != memory_domain:
            return False

        task_visual = str(task_payload.get("visualization_family", "")).strip()
        memory_visual = str(structured.get("visualization_family", "")).strip()
        if task_visual and memory_visual and task_visual != memory_visual:
            return False

        task_visual_contract = str(task_payload.get("visualization_contract_family", "")).strip()
        memory_visual_contract = str(structured.get("visualization_contract_family", "")).strip()
        if (
            task_visual_contract == "explicit_plot_labels_and_titles"
            and memory_visual_contract not in {"", "explicit_plot_labels_and_titles", "explicit_plot_bin_contract"}
        ):
            return False

        task_exception = str(task_payload.get("exception_contract_family", "")).strip()
        memory_exception = str(structured.get("exception_contract_family", "")).strip()
        explicit_exceptions = {"explicit_exception_contract"}
        if (
            task_exception in explicit_exceptions
            or memory_exception in explicit_exceptions
        ) and task_exception != memory_exception:
            return False
        return True

    @staticmethod
    def _boundary_layer(boundary):
        parts = [part for part in str(boundary or "").split("+") if part]
        collapse_boundaries = {
            "syntax_or_indentation_error",
            "missing_import_or_name_error",
            "missing_evaluation_result",
            "evaluator_error",
            "lexical_ban_violation",
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
        if any(part in collapse_boundaries for part in parts):
            return "collapse"
        if any(part in structural_boundaries for part in parts):
            return "structural"
        if any(part in semantic_boundaries for part in parts):
            return "semantic"
        if any(part in near_boundary_contracts for part in parts):
            return "near_boundary"
        if any(part in {"reuse_only_when_contract_matches", "calibrated_success"} for part in parts):
            return "success_boundary"
        if not parts:
            return "none"
        return "other"

    @classmethod
    def _boundary_alignment(cls, task_payload, structured):
        if not isinstance(task_payload, dict) or not isinstance(structured, dict):
            return "unknown"
        observed_boundary = str(task_payload.get("repair_target_failure_boundary", "")).strip()
        memory_boundary = str(structured.get("failure_boundary", "")).strip()
        if not observed_boundary or not memory_boundary:
            return "unknown"
        observed_parts = {part for part in observed_boundary.split("+") if part}
        memory_parts = {part for part in memory_boundary.split("+") if part}
        if observed_parts and memory_parts and observed_parts == memory_parts:
            return "exact"
        observed_layer = cls._boundary_layer(observed_boundary)
        memory_layer = cls._boundary_layer(memory_boundary)
        if observed_layer == memory_layer and observed_layer not in {"none", "other", "unknown"}:
            return "same_layer"
        near_pairs = {
            ("near_boundary", "semantic"),
            ("semantic", "near_boundary"),
            ("semantic", "structural"),
            ("structural", "semantic"),
        }
        if (observed_layer, memory_layer) in near_pairs:
            return "adjacent_layer"
        return "cross_layer"

    @staticmethod
    def _structure_score(task_payload, record):
        if not isinstance(task_payload, dict) or not isinstance(record, dict):
            return 0.0
        structured = record.get('s') or {}
        if not isinstance(structured, dict):
            return 0.0
        if not MemoryRetriever._code_contract_compatible(task_payload, structured):
            return 0.0

        checks = [
            ("task_domain_family", 0.3),
            ("abstract_signature", 0.45),
            ("input_family", 0.05),
            ("subject_family", 0.1),
            ("output_contract_family", 0.2),
            ("visualization_family", 0.1),
            ("visualization_contract_family", 0.15),
            ("return_slot_signature", 0.35),
            ("column_preservation_rule", 0.1),
            ("aggregation_rule", 0.2),
            ("plot_cardinality_rule", 0.15),
            ("axes_semantic_rule", 0.1),
            ("return_container_rule", 0.15),
            ("forbidden_patterns", 0.15),
            ("lexical_bans", 0.15),
            ("render_pattern_family", 0.1),
            ("exception_contract_family", 0.1),
        ]
        score = 0.0
        for key, weight in checks:
            lhs = str(task_payload.get(key, "")).strip()
            rhs = str(structured.get(key, "")).strip()
            if lhs and rhs and lhs == rhs:
                score += weight
        lhs_transform = str(task_payload.get("transform_family", "")).strip()
        rhs_transform = str(structured.get("action_type", "")).strip()
        if lhs_transform and rhs_transform and lhs_transform == rhs_transform:
            score += 0.25
        return min(1.0, score)

    @staticmethod
    def _reference_mode_label(memory_text):
        text = str(memory_text or "")
        if text.startswith("[Memory Retrieval]\ncore_match"):
            return "core_match"
        if text.startswith("[Memory Retrieval]\nseed_core_match"):
            return "seed_core_match"
        if text.startswith("[Memory Retrieval]\nreference_only"):
            return "reference_only"
        return ""

    @staticmethod
    def _format_structured_memory(structured):
        if not isinstance(structured, dict):
            return ""
        parts = [
            "[EMMA Structured Hint]",
            f"[task_type]\n{structured.get('task_type', '')}",
            f"[memory_level]\n{structured.get('memory_level', '')}",
            f"[input_family]\n{structured.get('input_family', '')}",
            f"[subject_family]\n{structured.get('subject_family', '')}",
            f"[column_preservation_rule]\n{structured.get('column_preservation_rule', '')}",
            f"[aggregation_rule]\n{structured.get('aggregation_rule', '')}",
            f"[transform_family]\n{structured.get('action_type', '')}",
            f"[output_contract_family]\n{structured.get('output_contract_family', '')}",
            f"[visualization_family]\n{structured.get('visualization_family', '')}",
            f"[visualization_contract_family]\n{structured.get('visualization_contract_family', '')}",
            f"[plot_cardinality_rule]\n{structured.get('plot_cardinality_rule', '')}",
            f"[axes_semantic_rule]\n{structured.get('axes_semantic_rule', '')}",
            f"[return_container_rule]\n{structured.get('return_container_rule', '')}",
            f"[forbidden_patterns]\n{structured.get('forbidden_patterns', '')}",
            f"[lexical_bans]\n{structured.get('lexical_bans', '')}",
            f"[render_pattern_family]\n{structured.get('render_pattern_family', '')}",
            f"[render_pattern_hint]\n{structured.get('render_pattern_hint', '')}",
            f"[exception_contract_family]\n{structured.get('exception_contract_family', '')}",
            f"[failure_boundary]\n{structured.get('failure_boundary', '')}",
            f"[verifier_check]\n{structured.get('verifier_check', '')}",
            f"[verifier_valid]\n{structured.get('verifier_valid', '')}",
            f"[verifier_failure_summary]\n{structured.get('verifier_failure_summary', '')}",
            f"[verifier_repair_constraint]\n{structured.get('verifier_repair_constraint', '')}",
            f"[reasoning_failure_pattern]\n{structured.get('reasoning_failure_pattern', '')}",
            f"[recompute_operator]\n{structured.get('recompute_operator', '')}",
            f"[disallowed_shortcut]\n{structured.get('disallowed_shortcut', '')}",
            f"[next_reasoning_move]\n{structured.get('next_reasoning_move', '')}",
            f"[proof_obligation]\n{structured.get('proof_obligation', '')}",
            f"[correction_rule]\n{structured.get('correction_rule', '')}",
            f"[success_contract_rule]\n{structured.get('success_contract_rule', '')}",
            f"[value_bias]\n{structured.get('value_bias', '')}",
            f"[evidence_topology_potential]\n{(structured.get('evidence') or {}).get('topology_potential', '')}",
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _memory_reference_heading(memory_text):
        text = str(memory_text or "")
        if "[value_bias]\npositive_reuse" in text or "[outcome]\nsuccess" in text:
            return "[Successful Experience]"
        if "[value_bias]\ncautionary_avoid" in text or "[outcome]\nfailure" in text:
            return "[Failure Experience]"
        return "[Memory Experience Reference]"

    def _render_memory_reference(self, label, memory_text):
        heading = self._memory_reference_heading(memory_text)
        return f"{label}\n{heading}\n{memory_text}"

    @staticmethod
    def _render_contract_hint_reference(memory_text):
        text = str(memory_text or "")
        if not text:
            return ""
        allowed_blocks = []
        keep_headers = {
            "[EMMA Structured Hint]",
            "[task_type]",
            "[memory_level]",
            "[input_family]",
            "[subject_family]",
            "[transform_family]",
            "[output_contract_family]",
            "[visualization_family]",
            "[visualization_contract_family]",
            "[column_preservation_rule]",
            "[matrix_operation_rule]",
            "[aggregation_rule]",
            "[plot_cardinality_rule]",
            "[axes_semantic_rule]",
            "[return_container_rule]",
            "[return_slot_signature]",
            "[plot_title_rule]",
            "[forbidden_patterns]",
            "[failure_boundary]",
            "[verifier_check]",
            "[verifier_valid]",
            "[verifier_failure_summary]",
            "[verifier_repair_constraint]",
            "[reasoning_failure_pattern]",
            "[recompute_operator]",
            "[disallowed_shortcut]",
            "[next_reasoning_move]",
            "[proof_obligation]",
            "[correction_rule]",
            "[success_contract_rule]",
            "[value_bias]",
        }
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        for chunk in chunks:
            header = chunk.splitlines()[0].strip()
            if header in keep_headers:
                allowed_blocks.append(chunk)
        if not allowed_blocks:
            allowed_blocks = [text]
        return "[Memory Retrieval]\ncontract_hint_only\n[Contract Hint Reference]\n" + "\n\n".join(allowed_blocks)

    @staticmethod
    def _empty_debug():
        return {
            "decision": "empty_memory",
            "best_idx": None,
            "max_sim": None,
            "max_combined": None,
            "best_structure_score": None,
            "best_q_score": None,
            "second_combined": None,
            "second_q_score": None,
            "q_margin": None,
            "combined_margin": None,
            "signature_match": False,
            "strict_contract_match": False,
            "trusted_structure_activation": False,
            "strong_priority_gap": False,
            "near_boundary_reference_blocked": False,
            "retrieval_threshold": None,
            "merge_threshold": None,
            "value_bias": "",
            "reason": "memory_bank_empty",
        }

    def _build_debug(
        self,
        *,
        decision: str,
        best_idx: int | None,
        max_sim: float,
        max_combined: float,
        best_structure_score: float,
        best_q_score: float,
        second_combined: float,
        second_q_score: float,
        q_margin: float,
        combined_margin: float,
        signature_match: bool,
        strict_contract_match: bool,
        trusted_structure_activation: bool,
        strong_priority_gap: bool,
        near_boundary_reference_blocked: bool,
        reason: str,
        value_bias: str = "",
    ):
        return {
            "decision": decision,
            "best_idx": best_idx,
            "max_sim": round(float(max_sim), 4) if best_idx is not None else None,
            "max_combined": round(float(max_combined), 4) if best_idx is not None else None,
            "best_structure_score": round(float(best_structure_score), 4) if best_idx is not None else None,
            "best_q_score": round(float(best_q_score), 4) if best_idx is not None else None,
            "second_combined": round(float(second_combined), 4) if second_combined >= 0.0 else None,
            "second_q_score": round(float(second_q_score), 4) if second_combined >= 0.0 else None,
            "q_margin": round(float(q_margin), 4) if best_idx is not None else None,
            "combined_margin": round(float(combined_margin), 4) if best_idx is not None else None,
            "signature_match": bool(signature_match),
            "strict_contract_match": bool(strict_contract_match),
            "trusted_structure_activation": bool(trusted_structure_activation),
            "strong_priority_gap": bool(strong_priority_gap),
            "near_boundary_reference_blocked": bool(near_boundary_reference_blocked),
            "retrieval_threshold": round(float(self.retrieval_threshold), 4),
            "merge_threshold": round(float(self.merge_threshold), 4),
            "value_bias": str(value_bias or "").strip(),
            "reason": reason,
        }

    def retrieve(self, current_z, memory_bank, task_payload=None):
        """
        根据当前任务意图 current_z，在 memory_bank 中检索最相似的经验。
        采用“只读/读写分离”架构，完美保护跨任务记忆的 Q 值不被污染。
        """
        if not memory_bank.records:
            print(colored("   [📭 记忆库为空] 开启零样本探索", "dark_grey"))
            return None, None, self._empty_debug()

        max_sim = -1.0
        max_combined = -1.0
        best_idx = None
        best_memory = None
        best_structure_score = 0.0
        best_is_trusted_strategy = False
        best_q_score = 0.0
        second_combined = -1.0
        second_q_score = 0.0

        # 将当前任务的特征向量展平，用于计算余弦相似度
        z_curr = np.squeeze(current_z)
        norm_curr = np.linalg.norm(z_curr)

        if norm_curr == 0:
            debug = self._empty_debug()
            debug["decision"] = "invalid_query"
            debug["reason"] = "zero_norm_query_embedding"
            debug["retrieval_threshold"] = round(float(self.retrieval_threshold), 4)
            debug["merge_threshold"] = round(float(self.merge_threshold), 4)
            return None, None, debug
        current_task_type = str((task_payload or {}).get("task_type", "")).strip() if isinstance(task_payload, dict) else ""
        strict_code_contract = bool(
            isinstance(task_payload, dict)
            and current_task_type == "code_generation"
            and (
                str(task_payload.get("return_slot_signature", "")).strip()
                or str(task_payload.get("plot_cardinality_rule", "")).strip()
                or str(task_payload.get("return_container_rule", "")).strip()
            )
        )
        # 遍历记忆库，寻找最高相似度的节点
        for i, record in enumerate(memory_bank.records):
            record_meta = record.get('meta', {}) if isinstance(record, dict) else {}
            record_task_type = str(record_meta.get('task_type', '')).strip()
            if current_task_type and record_task_type and current_task_type != record_task_type:
                continue
            if bool(record_meta.get('pruned', False)):
                continue
            structured = record.get('s') or {}
            strict_compatible = self._code_contract_compatible(task_payload, structured)
            reference_compatible = self._code_reference_compatible(task_payload, structured)
            if not reference_compatible:
                continue
            z_mem = np.squeeze(record['z'])
            norm_mem = np.linalg.norm(z_mem)
            
            if norm_mem == 0:
                continue
                
            # 计算余弦相似度 [-1.0, 1.0]
            sim = np.dot(z_curr, z_mem) / (norm_curr * norm_mem)
            structure_score = self._structure_score(task_payload, record)
            q_score = self._normalized_q(record)
            boundary_alignment = self._boundary_alignment(task_payload, structured)
            if current_task_type == "code_generation":
                if boundary_alignment == "exact":
                    structure_score = min(1.0, structure_score + 0.18)
                elif boundary_alignment == "same_layer":
                    structure_score = min(1.0, structure_score + 0.1)
                elif boundary_alignment == "adjacent_layer":
                    structure_score = min(1.0, structure_score + 0.03)
                elif boundary_alignment == "cross_layer":
                    structure_score = max(0.0, structure_score - 0.22)
                if not strict_compatible:
                    structure_score = max(0.0, structure_score - 0.12)
                    if (
                        str(task_payload.get("task_domain_family", "")).strip()
                        == str(structured.get("task_domain_family", "")).strip()
                        and str(task_payload.get("visualization_family", "")).strip()
                        == str(structured.get("visualization_family", "")).strip()
                    ):
                        structure_score = min(1.0, structure_score + 0.08)
            combined = (
                self.semantic_weight * float(sim)
                + self.structure_weight * structure_score
                + self.q_weight * q_score
            )
            is_trusted_strategy = (
                isinstance(structured, dict)
                and str(structured.get("memory_level", "")).strip() == "strategy"
                and bool(structured.get("trusted", False))
                and str(structured.get("value_bias", "")).strip() == "positive_reuse"
            )

            if combined > max_combined:
                second_combined = max_combined
                second_q_score = best_q_score
                max_sim = float(sim)
                max_combined = combined
                best_structure_score = structure_score
                best_idx = i
                best_is_trusted_strategy = is_trusted_strategy
                best_q_score = q_score
                best_memory = self._format_structured_memory(structured) or record.get('e') or ""
            elif combined > second_combined:
                second_combined = combined
                second_q_score = q_score


        signature_match = False
        if isinstance(task_payload, dict) and best_idx is not None:
            best_struct = memory_bank.records[best_idx].get('s', {}) if isinstance(memory_bank.records[best_idx], dict) else {}
            signature_match = (
                str(task_payload.get("abstract_signature", "")).strip()
                and str(task_payload.get("abstract_signature", "")).strip() == str(best_struct.get("abstract_signature", "")).strip()
            )
        strict_contract_match = False
        if isinstance(task_payload, dict) and best_idx is not None:
            best_struct = memory_bank.records[best_idx].get('s', {}) if isinstance(memory_bank.records[best_idx], dict) else {}
            if isinstance(best_struct, dict):
                strict_contract_keys = [
                    "return_slot_signature",
                    "plot_cardinality_rule",
                    "axes_semantic_rule",
                    "return_container_rule",
                    "render_pattern_family",
                ]
                strict_contract_match = True
                for key in strict_contract_keys:
                    lhs = str(task_payload.get(key, "")).strip()
                    rhs = str(best_struct.get(key, "")).strip()
                    if lhs and rhs and lhs != rhs:
                        strict_contract_match = False
                        break
        reference_contract_match = False
        if isinstance(task_payload, dict) and best_idx is not None:
            best_struct = memory_bank.records[best_idx].get('s', {}) if isinstance(memory_bank.records[best_idx], dict) else {}
            reference_contract_match = self._code_reference_compatible(task_payload, best_struct)

        trusted_structure_activation = (
            best_idx is not None
            and best_is_trusted_strategy
            and signature_match
            and best_structure_score >= 0.95
            and max_combined >= self.retrieval_threshold
        )
        q_margin = best_q_score - second_q_score if second_combined >= 0.0 else 1.0
        combined_margin = max_combined - second_combined if second_combined >= 0.0 else 1.0
        strong_priority_gap = (
            q_margin >= self.q_margin_threshold
            and combined_margin >= self.combined_margin_threshold
        )
        near_boundary_reference_blocked = (
            self._is_code_near_boundary_task(task_payload)
            and best_idx is not None
            and (
                (not signature_match and not reference_contract_match)
                or self._boundary_alignment(task_payload, best_struct) == "cross_layer"
            )
        )
        best_struct = memory_bank.records[best_idx].get('s', {}) if best_idx is not None and isinstance(memory_bank.records[best_idx], dict) else {}
        best_boundary_alignment = self._boundary_alignment(task_payload, best_struct)
        value_bias = str((best_struct or {}).get("value_bias", "")).strip() if isinstance(best_struct, dict) else ""

        should_hard_block_contract = (
            strict_code_contract
            and best_idx is not None
            and not strict_contract_match
            and (signature_match or best_structure_score >= 0.6)
        )
        if should_hard_block_contract:
            print(colored(f"   [🧱 合同隔离] 候选记忆未通过严格 contract 匹配 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "dark_grey"))
            return None, None, self._build_debug(
                decision="contract_blocked",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="strict_contract_mismatch",
                value_bias=value_bias,
            )
        if max_sim >= self.merge_threshold and signature_match and strong_priority_gap:
            # 🎯 任务 (读写模式)：返回经验 + 返回索引 (RL结算时会更新此Q值)
            print(colored(f"   [🎯 核心命中] 匹配到记忆 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f}, Q差: {q_margin:.2f}, 排序差: {combined_margin:.2f}) ", "green", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\ncore_match", best_memory)
            return m_ctx, best_idx, self._build_debug(
                decision="core_match",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="merge_threshold_and_signature_match" if best_boundary_alignment in {"unknown", "exact", "same_layer"} else "merge_threshold_but_cross_layer",
                value_bias=value_bias,
            )
        elif (
            current_task_type == "code_generation"
            and best_idx is not None
            and signature_match
            and strict_contract_match
            and max_combined >= self.retrieval_threshold
            and best_boundary_alignment != "cross_layer"
        ):
            print(colored(f"   [🔁 同签名修补] 同签名高置信候选放宽 Q-gap 阻塞 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "green", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\ncore_match", best_memory)
            return m_ctx, best_idx, self._build_debug(
                decision="core_match",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="same_signature_high_confidence_override",
                value_bias=value_bias,
            )
        elif trusted_structure_activation and strong_priority_gap:
            print(colored(f"   [🌱 Seed 激活] 可信策略边作为核心结构命中 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f}, Q差: {q_margin:.2f}, 排序差: {combined_margin:.2f})", "green", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\nseed_core_match", best_memory)
            return m_ctx, best_idx, self._build_debug(
                decision="seed_core_match",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="trusted_strategy_activation",
                value_bias=value_bias,
            )
        elif near_boundary_reference_blocked and max_combined >= self.retrieval_threshold:
            print(colored(f"   [🪫 边界降权] near-boundary 任务禁止跨任务正经验注入 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "dark_grey"))
            return None, None, self._build_debug(
                decision="blocked",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="near_boundary_reference_blocked",
                value_bias=value_bias,
            )
        elif (
            current_task_type == "code_generation"
            and best_idx is not None
            and reference_contract_match
            and not strict_contract_match
            and max_combined >= max(0.58, self.retrieval_threshold - 0.12)
            and best_boundary_alignment != "cross_layer"
        ):
            print(colored(f"   [🪶 合同提示] 放宽为只读 contract hint (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "blue", attrs=['bold']))
            m_ctx = self._render_contract_hint_reference(best_memory)
            return m_ctx, None, self._build_debug(
                decision="contract_hint_only",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="same_domain_visual_family_contract_hint",
                value_bias=value_bias,
            )
        elif (
            current_task_type == "code_generation"
            and best_idx is not None
            and strict_contract_match
            and not signature_match
            and max_combined >= max(0.6, self.retrieval_threshold - 0.15)
            and best_boundary_alignment != "cross_layer"
        ):
            print(colored(f"   [🧭 合同近邻] strict contract 已匹配，放宽为低门槛只读参考 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "blue", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\nreference_only", best_memory)
            return m_ctx, None, self._build_debug(
                decision="reference_only",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="strict_contract_near_neighbor_reference",
                value_bias=value_bias,
            )
        elif max_combined >= self.retrieval_threshold and strong_priority_gap and best_boundary_alignment != "cross_layer":
            # 💡 跨任务(只读模式)：返回经验 + 强制隐藏索引 None！
            # 这样 run_episode.py 里的 RL 结算面板根本拿不到索引，完美保护旧记忆 Q 值！
            print(colored(f"   [💡 跨任务参考] 借用结构经验 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f}, Q差: {q_margin:.2f}, 排序差: {combined_margin:.2f})", "blue", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\nreference_only", best_memory)
            return m_ctx, None, self._build_debug(
                decision="reference_only",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="combined_threshold_met_without_core_path",
                value_bias=value_bias,
            )
        elif max_combined >= self.retrieval_threshold and best_boundary_alignment == "cross_layer":
            print(colored(f"   [🧱 边界隔离] 候选记忆跨越 failure layer，禁止注入 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "dark_grey"))
            return None, None, self._build_debug(
                decision="blocked",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="cross_boundary_layer_blocked",
                value_bias=value_bias,
            )
        elif (
            current_task_type == "closed_ended_reasoning"
            and best_idx is not None
            and max_combined >= self.retrieval_threshold
            and str((best_struct or {}).get("outcome", "")).strip() == "failure"
            and str((best_struct or {}).get("verifier_check", "")).strip()
        ):
            print(colored(f"   [🧪 Verifier 只读参考] 注入可审计失败记忆 (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f}, Q差: {q_margin:.2f})", "blue", attrs=['bold']))
            m_ctx = self._render_memory_reference("[Memory Retrieval]\nreference_only", best_memory)
            return m_ctx, None, self._build_debug(
                decision="verifier_reference_only",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="closed_ended_verifier_failure_reference",
                value_bias=value_bias,
            )
        elif max_combined >= self.retrieval_threshold:
            print(colored(f"   [⚖️ 排序未拉开] 候选记忆未形成足够 Q / 排序差距 (Q差: {q_margin:.2f}, 排序差: {combined_margin:.2f})", "dark_grey"))
            return None, None, self._build_debug(
                decision="blocked",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="priority_gap_too_small",
                value_bias=value_bias,
            )

        else:
            print(colored(f"   [🚧 未命中] 组合分 {max_combined:.4f} < {self.retrieval_threshold:.4f} (相似度: {max_sim:.4f}, 结构分: {best_structure_score:.2f})", "dark_grey"))
            return None, None, self._build_debug(
                decision="miss",
                best_idx=best_idx,
                max_sim=max_sim,
                max_combined=max_combined,
                best_structure_score=best_structure_score,
                best_q_score=best_q_score,
                second_combined=second_combined,
                second_q_score=second_q_score,
                q_margin=q_margin,
                combined_margin=combined_margin,
                signature_match=signature_match,
                strict_contract_match=strict_contract_match,
                trusted_structure_activation=trusted_structure_activation,
                strong_priority_gap=strong_priority_gap,
                near_boundary_reference_blocked=near_boundary_reference_blocked,
                reason="combined_score_below_threshold",
                value_bias=value_bias,
            )

    def assemble_prompt(self, current_context, m_ctx, task_type=None):
        """
        将战术面板（当前状态）与战略指南（历史记忆）进行最终组装
        """
        if m_ctx:
            if task_type == "code_generation":
                prompt = (
                    "[Memory Knowledge Reference]:\n"
                    "The retrieved memory is prior experience, not a script to imitate.\n"
                    "Use it only as abstract decision reference: input family, transform family, output contract, failure boundary, success contract, and value bias.\n"
                    "Do NOT copy wording, starter code, function bodies, markdown, bullets, indentation, or any surface template from memory.\n"
                    "Solve the current task from the current task description and starter header only.\n"
                    "The current task's return signature, output slot order, and evaluator-facing container contract override any retrieved memory.\n"
                    "If the reference is a Successful Experience, preserve the relevant contract and strategy pattern without copying its surface form.\n"
                    "If the reference is a Failure Experience, treat its failure_boundary as a boundary to avoid and use its correction_rule as repair knowledge when relevant.\n"
                    "If the reference exposes a success_contract_rule, use it to verify the final implementation contract before submission.\n"
                    "If the reference exposes a render_pattern_family, use that only when it truly matches the current task contract.\n"
                    "If the current task provides lexical guardrails or forbidden patterns, treat them as hard blockers and remove any banned pattern before submitting code.\n"
                    "Before emitting code, perform one final self-check against failure_boundary, correction_rule, and success_contract_rule.\n\n"
                    f"{m_ctx}\n\n[CURRENT Environment & State]:\n{current_context}"
                )
            elif task_type == "closed_ended_reasoning":
                prompt = (
                    "[Memory Knowledge Reference]:\n"
                    "The retrieved memory is prior experience, not answer text.\n"
                    "Use it only as abstract reference: input family, output contract, failure boundary, verifier evidence, correction rule, proof obligation, success contract, and value bias.\n"
                    "If the reference contains verifier_failure_summary or verifier_repair_constraint, treat it as an environment-side hard constraint for the current repair.\n"
                    "If the reference contains disallowed_shortcut, recompute_operator, next_reasoning_move, or proof_obligation, use those fields to choose the reasoning path before selecting Exact Answer.\n"
                    "Never quote, restate, or copy memory text into the final answer.\n"
                    "Your final output must still follow the benchmark's declared answer format exactly.\n\n"
                    f"{m_ctx}\n\n[CURRENT Environment & State]:\n{current_context}"
                )
            else:
                prompt = f"{m_ctx}\n\n[CURRENT Environment & State]:\n{current_context}"
        else:
            prompt = (
                "[Memory Knowledge Reference]:\n"
                "None (no prior successful or failed experience was retrieved; rely on zero-shot reasoning from the current task only.)\n\n"
                f"[CURRENT Environment & State]:\n{current_context}"
            )
        return prompt
