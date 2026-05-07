from __future__ import annotations

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class WebArenaAdapter(BenchmarkAdapter):
    benchmark_name = "webarena"

    def setup(self) -> None:
        raise NotImplementedError(
            "WebArena adapter skeleton created. Next step is to bind browser state serialization and Docker/browser task harness."
        )

    def reset_task(self, index: int | None = None) -> ResetResult:
        raise NotImplementedError

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def force_finish(self) -> StepResult:
        raise NotImplementedError
