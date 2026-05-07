from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    task_id: str
    instruction: str
    task_type: str = ""
    task_description: str = ""
    goal_repr: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal_repr:
            self.goal_repr = self.task_description or self.instruction


@dataclass
class ResetResult:
    task: TaskSpec
    observation: str
    state_repr: str = ""
    candidate_actions: list[str] = field(default_factory=list)
    valid_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.state_repr:
            self.state_repr = self.observation
        if not self.candidate_actions and self.valid_actions:
            self.candidate_actions = list(self.valid_actions)
        if not self.valid_actions and self.candidate_actions:
            self.valid_actions = list(self.candidate_actions)


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    success: bool
    state_repr: str = ""
    candidate_actions: list[str] = field(default_factory=list)
    failure_signal: str = ""
    terminal_status: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    valid_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.state_repr:
            self.state_repr = self.observation
        if not self.candidate_actions and self.valid_actions:
            self.candidate_actions = list(self.valid_actions)
        if not self.valid_actions and self.candidate_actions:
            self.valid_actions = list(self.candidate_actions)
        if not self.terminal_status:
            if self.done and self.success:
                self.terminal_status = "success"
            elif self.done:
                self.terminal_status = "failure"
            else:
                self.terminal_status = "running"


class BenchmarkAdapter(ABC):
    benchmark_name: str = "unknown"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        self.config = config
        self.traj_dir = traj_dir

    def setup(self) -> None:
        return None

    @abstractmethod
    def reset_task(self, index: int | None = None) -> ResetResult:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def force_finish(self) -> StepResult:
        raise NotImplementedError

    def normalize_action(self, raw_action: str) -> str:
        return (raw_action or "").strip()

    def get_valid_actions(self) -> list[str]:
        return []

    def task_semantic_text(self, task: TaskSpec) -> str:
        parts = []
        if task.task_type:
            parts.append(f"[task_type={task.task_type}]")
        if task.goal_repr:
            parts.append(f"[goal]\n{task.goal_repr}")
        elif task.task_description:
            parts.append(task.task_description)
        elif task.instruction:
            parts.append(task.instruction)
        return "\n".join(part for part in parts if part)

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
        parts = [self.task_semantic_text(task)]
        state_block = state_repr or observation
        if state_block:
            parts.append(f"[state]\n{state_block}")
        if valid_actions:
            preview = "\n".join(valid_actions[:20])
            parts.append(f"[candidate_actions]\n{preview}")
        if failure_signal:
            parts.append(f"[failure_signal]\n{failure_signal}")
        if history_lines:
            parts.append(f"[recent_history]\n" + "\n".join(history_lines[-4:]))
        return "\n\n".join(part for part in parts if part)

    def seed_memories(self) -> list[dict[str, Any]]:
        return []

    def route_probe(self, task: TaskSpec, action: str) -> dict[str, Any] | None:
        return None

    def route_prompt(self, task: TaskSpec, observation: str) -> str:
        return f"{self.task_semantic_text(task)}\n\n{observation}".strip()

    def route_hint(self, task: TaskSpec, observation: str) -> dict[str, Any]:
        return {}

    def build_repair_prompt(self, task: TaskSpec, last_trace: dict[str, Any]) -> str:
        return ""

    def history_entry(self, action: str, observation: str, step_result: StepResult) -> str:
        return f"> {action}\n< {observation}"

    def close(self) -> None:
        return None
