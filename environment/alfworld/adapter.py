from __future__ import annotations

from typing import Any

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class ALFWorldAdapter(BenchmarkAdapter):
    benchmark_name = "alfworld"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.env = None
        self.current_task = None
        self.current_valid_actions: list[str] = []
        self.current_observation = ""
        self._episode_counter = 0

    def setup(self) -> None:
        from exp.modules.env_wrapper import ALFWorldEnvWrapper

        difficulty = self.config.get("runner", {}).get("difficulty", "hard")
        self.env = ALFWorldEnvWrapper(difficulty=difficulty)

    def reset_task(self, index: int | None = None) -> ResetResult:
        observation, task, valid_actions = self.env.reset()
        self._episode_counter += 1
        self.current_valid_actions = valid_actions
        self.current_observation = observation
        task_type = self._infer_task_type(task)
        self.current_task = TaskSpec(
            task_id=f"alfworld[{self._episode_counter}]",
            instruction=task,
            task_type=task_type,
            task_description=task,
            metadata={"valid_actions": valid_actions},
        )
        return ResetResult(task=self.current_task, observation=observation, valid_actions=valid_actions)

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-6:]) if history_lines else "None"
        valid_actions = "\n".join(f"- {action}" for action in self.current_valid_actions[:80])
        return (
            f"Goal task:\n{task.instruction}\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Recent action history:\n{history_block}\n\n"
            f"Available actions:\n{valid_actions}\n\n"
            "Output exactly one action from the available actions list."
        )

    def step(self, action: str) -> StepResult:
        next_obs, reward, done, trace, next_valid_actions = self.env.step(action)
        self.current_observation = next_obs
        self.current_valid_actions = next_valid_actions
        success = bool(trace.get("is_success", False))
        return StepResult(
            observation=next_obs,
            reward=float(reward),
            done=bool(done),
            success=success,
            info={"trace": trace, "valid_actions": next_valid_actions},
            valid_actions=next_valid_actions,
        )

    def force_finish(self) -> StepResult:
        return StepResult(
            observation=self.current_observation,
            reward=0.0,
            done=True,
            success=False,
            info={"forced_terminate": True, "valid_actions": self.current_valid_actions},
            valid_actions=self.current_valid_actions,
        )

    def get_valid_actions(self) -> list[str]:
        return list(self.current_valid_actions)

    @staticmethod
    def _infer_task_type(task_text: str) -> str:
        text = (task_text or "").lower()
        if "put two" in text:
            return "alfworld_put_two"
        if "heat" in text:
            return "alfworld_heat"
        if "cool" in text:
            return "alfworld_cool"
        if "clean" in text:
            return "alfworld_clean"
        if "examine" in text or "look at" in text:
            return "alfworld_examine"
        if "put" in text or "place" in text:
            return "alfworld_put"
        return "alfworld_general"
