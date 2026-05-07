from __future__ import annotations

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class OSWorldAdapter(BenchmarkAdapter):
    benchmark_name = "osworld"

    def setup(self) -> None:
        raise NotImplementedError(
            "OSWorld adapter skeleton created. Next step is to bind desktop state serialization and local/remote system task harness."
        )

    def reset_task(self, index: int | None = None) -> ResetResult:
        raise NotImplementedError

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def force_finish(self) -> StepResult:
        raise NotImplementedError
