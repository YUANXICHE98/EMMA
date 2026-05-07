from __future__ import annotations

import json
import os
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class LifelongAgentBenchAdapter(BenchmarkAdapter):
    benchmark_name = "lifelong_agent_bench"
    _SUPPORTED_TASKS = {"knowledge_graph"}
    _VALID_ACTIONS = [
        "Action: get_relations(...)",
        "Action: get_neighbors(..., ...)",
        "Action: intersection(#i, #j)",
        "Action: get_attributes(#i)",
        "Action: argmax(#i, ...)",
        "Action: argmin(#i, ...)",
        "Action: count(#i)",
        "Final Answer: #i",
    ]

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.runner_cfg = config.get("runner", {})
        self.execution_mode = str(self.runner_cfg.get("execution_mode", "direct")).strip().lower()
        self.lab_repo_path: Path | None = None
        self.lab = None
        self.task_interface = None
        self.sample_index_list: list[Any] = []
        self.current_session = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self._session_completed = False

    def setup(self) -> None:
        task_name = self._task_name()
        if task_name not in self._SUPPORTED_TASKS:
            supported = ", ".join(sorted(self._SUPPORTED_TASKS))
            raise NotImplementedError(
                f"LifelongAgentBench adapter currently supports task_name in {{{supported}}}, got `{task_name}`."
            )

        self.lab_repo_path = self._resolve_repo_path()
        self.lab = self._import_lab_modules(self.lab_repo_path)

        if self.execution_mode == "client":
            self.task_interface = self._build_client()
        elif self.execution_mode == "direct":
            self.task_interface = self._build_direct_task()
        else:
            raise ValueError(
                f"Unsupported LifelongAgentBench execution_mode `{self.execution_mode}`. Use `direct` or `client`."
            )

        self.sample_index_list = list(self.task_interface.get_sample_index_list())
        if not self.sample_index_list:
            raise RuntimeError("LifelongAgentBench adapter loaded zero samples from the official task.")

    def reset_task(self, index: int | None = None) -> ResetResult:
        if not self.sample_index_list:
            raise RuntimeError("LifelongAgentBench adapter has no loaded samples.")

        sample_position = 0 if index is None else index
        if sample_position < 0 or sample_position >= len(self.sample_index_list):
            raise IndexError(
                f"LifelongAgentBench sample index out of range: {sample_position} / {len(self.sample_index_list)}"
            )

        sample_index = self.sample_index_list[sample_position]
        self.current_session = self.lab.Session(
            task_name=self._official_task_name(),
            sample_index=sample_index,
        )
        self.task_interface.reset(self.current_session)
        self._session_completed = False

        opening_user_message = self._last_user_message(self.current_session)
        question_message = self._question_message(self.current_session)
        self.current_task = TaskSpec(
            task_id=f"lifelong_agent_bench[{self._task_name()}][{sample_index}]",
            instruction=question_message or opening_user_message,
            task_type=f"lifelong_agent_bench/{self._task_name()}",
            task_description=question_message or opening_user_message,
            goal_repr=self._goal_repr(self.current_session),
            metadata={
                "benchmark": self.benchmark_name,
                "task_name": self._task_name(),
                "sample_index": sample_index,
                "execution_mode": self.execution_mode,
                "official_server_url": self.runner_cfg.get("server_url", ""),
                "abstract_goal": self._goal_repr(self.current_session),
                "entity_hints": self._entity_hints(self.current_session),
                "long_horizon_goal": "Accumulate verified knowledge across sessions without reusing stale or invalid facts.",
            },
        )
        self.current_observation = self._render_observation(self.current_session)
        return ResetResult(
            task=self.current_task,
            observation=self.current_observation,
            state_repr=self._state_repr(self.current_session),
            valid_actions=self.get_valid_actions(),
        )

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-6:]) if history_lines else "None"
        return (
            f"Benchmark: LifelongAgentBench\n"
            f"Task: {task.metadata.get('task_name', self._task_name())}\n"
            f"Sample: {task.metadata.get('sample_index')}\n\n"
            "You are interacting with the official benchmark environment.\n"
            "Output exactly one official action string and nothing else.\n\n"
            "Valid output formats:\n"
            "- Action: get_relations(argument)\n"
            "- Action: get_neighbors(argument, relation)\n"
            "- Action: intersection(#i, #j)\n"
            "- Action: get_attributes(#i)\n"
            "- Action: argmax(#i, attribute)\n"
            "- Action: argmin(#i, attribute)\n"
            "- Action: count(#i)\n"
            "- Final Answer: #i\n\n"
            "Policy constraints:\n"
            "1. Use exact entity names, relation names, attribute names, and variable references shown by the environment.\n"
            "2. Do not explain your reasoning.\n"
            "3. If the environment returns an execution error, repair the action format or arguments instead of repeating it unchanged.\n"
            "4. Only output `Final Answer: #i` when the referenced variable is the intended final answer.\n\n"
            f"Current environment transcript:\n{observation}\n\n"
            f"Recent action history:\n{history_block}"
        )

    def task_semantic_text(self, task: TaskSpec) -> str:
        metadata = task.metadata or {}
        parts = [
            f"[task_type={task.task_type}]",
            f"[goal]\n{task.goal_repr}",
        ]
        long_horizon_goal = str(metadata.get("long_horizon_goal", "")).strip()
        if long_horizon_goal:
            parts.append(f"[long_horizon_goal]\n{long_horizon_goal}")
        entity_hints = metadata.get("entity_hints") or []
        if entity_hints:
            parts.append("[entity_hints]\n" + ", ".join(str(item) for item in entity_hints))
        return "\n\n".join(part for part in parts if part)

    def step(self, action: str) -> StepResult:
        if self.current_session is None:
            raise RuntimeError("LifelongAgentBench step called before reset_task.")

        self.current_session.chat_history.inject(
            {
                "role": self.lab.Role.AGENT,
                "content": action,
            }
        )
        self.task_interface.interact(self.current_session)
        done = self.current_session.sample_status != self.lab.SampleStatus.RUNNING
        info = self._session_info(self.current_session)
        reward = 0.0
        success = False

        if done:
            self._complete_current_session()
            info = self._session_info(self.current_session)
            reward = float(info.get("f1_score", 0.0))
            success = bool(info.get("exact_match", False))

        self.current_observation = self._render_observation(self.current_session)
        failure_signal = self._failure_signal(self.current_session, info)
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            success=success,
            state_repr=self._state_repr(self.current_session),
            failure_signal=failure_signal,
            info=info,
            valid_actions=self.get_valid_actions(),
        )

    def force_finish(self) -> StepResult:
        if self.current_session is None:
            return StepResult(
                observation="LifelongAgentBench episode terminated before initialization.",
                reward=0.0,
                done=True,
                success=False,
                info={"forced_terminate": True},
                valid_actions=self.get_valid_actions(),
            )

        if self.current_session.sample_status == self.lab.SampleStatus.RUNNING:
            self.current_session.sample_status = self.lab.SampleStatus.AGENT_CONTEXT_LIMIT
            self.current_session.finish_reason = "MemRL max_steps reached before the agent produced a final answer."
            self.current_session.task_output = {"answer": None}

        self._complete_current_session()
        self.current_observation = self._render_observation(self.current_session)
        info = self._session_info(self.current_session)
        return StepResult(
            observation=self.current_observation,
            reward=float(info.get("f1_score", 0.0)),
            done=True,
            success=bool(info.get("exact_match", False)),
            state_repr=self._state_repr(self.current_session),
            failure_signal=self._failure_signal(self.current_session, info),
            info={**info, "forced_terminate": True},
            valid_actions=self.get_valid_actions(),
        )

    def normalize_action(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if action.lower().startswith("```"):
            lines = [line for line in action.splitlines() if not line.strip().startswith("```")]
            action = "\n".join(lines).strip()
        if not action:
            return ""

        first_line = action.splitlines()[0].strip()
        if first_line.lower().startswith("action:"):
            return f"Action: {first_line.split(':', 1)[1].strip()}"
        if first_line.lower().startswith("final answer:"):
            return f"Final Answer: {first_line.split(':', 1)[1].strip()}"
        if "(" in first_line and ")" in first_line:
            return f"Action: {first_line}"
        return first_line

    def get_valid_actions(self) -> list[str]:
        return list(self._VALID_ACTIONS)

    def close(self) -> None:
        if self.task_interface is not None:
            try:
                self.task_interface.release()
            except Exception:
                return

    def _build_direct_task(self):
        if self.lab_repo_path is None:
            raise RuntimeError("Direct LifelongAgentBench execution requires a local lab_repo_path.")

        task_name = self._task_name()
        if task_name != "knowledge_graph":
            raise NotImplementedError(f"Direct LifelongAgentBench adapter does not yet support task `{task_name}`.")

        chat_history_item_dict_path = self._resolve_repo_relative_path(
            self.runner_cfg.get("chat_history_item_dict_path", "chat_history_items/standard/knowledge_graph.json")
        )
        data_file_path = self._resolve_repo_relative_path(
            self.runner_cfg.get(
                "data_file_path",
                "data/v0303/knowledge_graph/processed/grailqa/"
                "v0417_tl2sc50_tl3sc50_tl4sc50_tl5sc50_tl6sc50_tl7sc50_tl8sc50_tl9sc46/entry_dict.json",
            )
        )
        ontology_dir_path = self._resolve_repo_relative_path(
            self.runner_cfg.get("ontology_dir_path", "data/v0121/knowledge_graph/ontology")
        )

        factory = self.lab.ChatHistoryItemFactory(str(chat_history_item_dict_path))
        return self.lab.KnowledgeGraph(
            task_name=self._official_task_name(),
            chat_history_item_factory=factory,
            sparql_url=self.runner_cfg.get("sparql_url", "http://127.0.0.1:3001/sparql"),
            ontology_dir_path=str(ontology_dir_path),
            data_file_path=str(data_file_path),
            max_round=int(self.runner_cfg.get("task_max_round", 15)),
        )

    def _build_client(self):
        server_url = str(self.runner_cfg.get("server_url", "http://127.0.0.1:8000/api")).strip()
        request_timeout = int(self.runner_cfg.get("request_timeout", 120))
        return self.lab.TaskClient(server_address=server_url, request_timeout=request_timeout)

    def _complete_current_session(self) -> None:
        if self.current_session is None or self._session_completed:
            return
        self.task_interface.complete(self.current_session)
        self._session_completed = True

    def _session_info(self, session) -> dict[str, Any]:
        detail_dict = session.evaluation_record.detail_dict or {}
        exact_match = session.evaluation_record.outcome == self.lab.SessionEvaluationOutcome.CORRECT
        return {
            "sample_status": str(session.sample_status),
            "finish_reason": session.finish_reason,
            "task_output": session.task_output,
            "evaluation_outcome": str(session.evaluation_record.outcome),
            "exact_match": exact_match,
            "f1_score": float(detail_dict.get("f1_score", 0.0) or 0.0),
            "executable_flag": bool(detail_dict.get("executable_flag", False)),
        }

    def _render_observation(self, session) -> str:
        role_map = {
            self.lab.Role.USER: "USER",
            self.lab.Role.AGENT: "AGENT",
        }
        chat_text = session.chat_history.get_value_str(role_map, start_index=0, end_index=session.chat_history.get_value_length())
        parts = [
            f"[sample_status] {session.sample_status}",
            chat_text,
        ]
        if session.finish_reason:
            parts.append(f"[finish_reason] {session.finish_reason}")
        if session.evaluation_record.outcome != self.lab.SessionEvaluationOutcome.UNSET:
            parts.append(f"[evaluation_outcome] {session.evaluation_record.outcome}")
            if session.evaluation_record.detail_dict:
                parts.append("[evaluation_detail] " + json.dumps(session.evaluation_record.detail_dict, ensure_ascii=False))
        return "\n\n".join(part for part in parts if part)

    def _goal_repr(self, session) -> str:
        question_message = self._question_message(session)
        entity_hints = self._entity_hints(session)
        parts = [
            "Solve the current knowledge-graph session by turning entity hints into validated relations, variables, and a final executable answer.",
        ]
        if question_message:
            parts.append(f"current_session_question={question_message}")
        if entity_hints:
            parts.append("entity_hints=" + ", ".join(entity_hints))
        parts.append(
            "long_horizon_requirement=reuse only previously validated facts and suppress stale, invalid, or repeated failed queries."
        )
        return "\n".join(parts)

    def _state_repr(self, session) -> str:
        entity_hints = self._entity_hints(session)
        verified_facts = self._verified_fact_lines(session)
        known_variables = self._known_variables(session)
        parts = [
            f"sample_status={session.sample_status}",
            f"task_name={self._task_name()}",
        ]
        question_message = self._question_message(session)
        if question_message:
            parts.append(f"question={question_message}")
        if entity_hints:
            parts.append("known_entities=" + ", ".join(entity_hints))
        if known_variables:
            parts.append("known_variables=" + ", ".join(known_variables))
        if verified_facts:
            parts.append("validated_facts=" + " || ".join(verified_facts[-4:]))
        else:
            parts.append("validated_facts=none_yet")
        if session.finish_reason:
            parts.append(f"latest_failure_or_stop={session.finish_reason}")
        if session.task_output:
            parts.append("task_output=" + json.dumps(session.task_output, ensure_ascii=False))
        return "\n".join(parts)

    def _failure_signal(self, session, info: dict[str, Any]) -> str:
        sample_status = str(info.get("sample_status", "") or session.sample_status).strip().lower()
        finish_reason = str(info.get("finish_reason", "") or session.finish_reason or "").strip().lower()
        executable_flag = bool(info.get("executable_flag", False))
        exact_match = bool(info.get("exact_match", False))

        if sample_status == "running":
            return ""
        if sample_status == "task_environment_error":
            if "query failed" in finish_reason:
                return "environment_query_failed"
            return "task_environment_error"
        if sample_status == "agent_validation_failed":
            if "cannot find the pattern of action" in finish_reason:
                return "invalid_action_format"
            return "invalid_action"
        if sample_status == "agent_context_limit":
            return "agent_context_limit"
        if sample_status == "task_limit_reached":
            return "repeated_invalid_query_or_round_limit"
        if sample_status == "completed":
            if exact_match:
                return ""
            if not executable_flag:
                return "non_executable_final_answer"
            return "wrong_final_answer_or_stale_fact"
        if sample_status:
            return sample_status
        return ""

    def _entity_hints(self, session) -> list[str]:
        question_message = self._question_message(session)
        if "Entities:" not in question_message:
            return []
        entity_text = question_message.split("Entities:", 1)[1].strip()
        entity_text = entity_text.strip("[]")
        if not entity_text:
            return []
        raw_items = [item.strip().strip("'\"") for item in entity_text.split(",")]
        return [item for item in raw_items if item]

    def _verified_fact_lines(self, session) -> list[str]:
        facts: list[str] = []
        for idx in range(session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role != self.lab.Role.USER:
                continue
            content = str(item.content)
            if content.startswith("Question: ") or content.startswith("You are an intelligent agent"):
                continue
            if content == "OK.":
                continue
            facts.append(content)
        return facts

    def _known_variables(self, session) -> list[str]:
        variable_sources: list[str] = []
        variable_sources.extend(self._verified_fact_lines(session))
        for idx in range(session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role == self.lab.Role.AGENT:
                variable_sources.append(str(item.content))
        variables = set(re.findall(r"#\d+", "\n".join(variable_sources)))
        return sorted(variables, key=lambda item: int(item[1:]))

    def _question_message(self, session) -> str:
        for idx in range(session.chat_history.get_value_length() - 1, -1, -1):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role == self.lab.Role.USER and str(item.content).startswith("Question: "):
                return str(item.content)
        return ""

    def _last_user_message(self, session) -> str:
        for idx in range(session.chat_history.get_value_length() - 1, -1, -1):
            item = session.chat_history.get_item_deep_copy(idx)
            if item.role == self.lab.Role.USER:
                return str(item.content)
        return ""

    def _official_task_name(self):
        task_name = self._task_name()
        if task_name == "knowledge_graph":
            return self.lab.TaskName.KNOWLEDGE_GRAPH
        raise NotImplementedError(f"Unsupported LifelongAgentBench task_name `{task_name}`.")

    def _task_name(self) -> str:
        return str(self.runner_cfg.get("task_name", "knowledge_graph")).strip().lower()

    def _resolve_repo_path(self) -> Path | None:
        configured = (
            self.runner_cfg.get("lab_repo_path")
            or self.config.get("lab_repo_path")
            or ""
        )
        env_override = (
            self.runner_cfg.get("lab_repo_env_var")
            or self.config.get("lab_repo_env_var")
            or "MEMRL_LAB_REPO_PATH"
        )
        if configured:
            path = Path(str(configured)).expanduser()
            return path.resolve()
        env_path = str(os.environ.get(str(env_override), "")).strip()
        if env_path:
            return Path(env_path).expanduser().resolve()
        return None

    def _resolve_repo_relative_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute():
            return path
        if self.lab_repo_path is None:
            raise RuntimeError(
                f"LifelongAgentBench relative path `{raw_path}` requires runner.lab_repo_path or MEMRL_LAB_REPO_PATH."
            )
        return (self.lab_repo_path / path).resolve()

    def _import_lab_modules(self, repo_path: Path | None):
        if repo_path is not None:
            if not repo_path.exists():
                raise FileNotFoundError(
                    f"LifelongAgentBench repo path does not exist: {repo_path}. "
                    "Clone the official repo locally and point runner.lab_repo_path to it."
                )
            repo_str = str(repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        try:
            general = import_module("src.typings.general")
            session_module = import_module("src.typings.session")
            status_module = import_module("src.typings.status")
            task_client_module = import_module("src.tasks.client")
            kg_module = import_module("src.tasks.instance.knowledge_graph")
            chat_factory_module = import_module("src.factories.chat_history_item.online.chat_history_item_factory")
        except Exception as exc:
            repo_hint = f" repo_path={repo_path}" if repo_path is not None else ""
            raise RuntimeError(
                "Failed to import official LifelongAgentBench modules. "
                "Make sure the repo is cloned locally and its Python dependencies are installed."
                " Then set runner.lab_repo_path or MEMRL_LAB_REPO_PATH."
                f"{repo_hint}"
            ) from exc

        class _LabNamespace:
            Role = general.Role
            TaskName = general.TaskName
            Session = session_module.Session
            SessionEvaluationOutcome = session_module.SessionEvaluationOutcome
            SampleStatus = status_module.SampleStatus
            TaskClient = task_client_module.TaskClient
            KnowledgeGraph = kg_module.KnowledgeGraph
            ChatHistoryItemFactory = chat_factory_module.ChatHistoryItemFactory

        return _LabNamespace
