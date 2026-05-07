from __future__ import annotations

from typing import Any

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class ScienceWorldAdapter(BenchmarkAdapter):
    benchmark_name = "scienceworld"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.runner_cfg = config.get("runner", {})
        self.env = None
        self.task_plan: list[tuple[str, int]] = []
        self.current_task: TaskSpec | None = None
        self.current_valid_actions: list[str] = []
        self.current_observation = ""

    def setup(self) -> None:
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError as exc:
            raise RuntimeError(
                "ScienceWorld adapter requires the `scienceworld` package. "
                "Use environment/.venv-scienceworld/bin/python to run this benchmark."
            ) from exc

        env_step_limit = int(self.runner_cfg.get("env_step_limit", self.runner_cfg.get("max_steps_per_episode", 100)))
        server_path = (self.runner_cfg.get("server_path") or "").strip() or None
        self.env = ScienceWorldEnv(serverPath=server_path, envStepLimit=env_step_limit)
        self.task_plan = self._build_task_plan()
        if not self.task_plan:
            raise RuntimeError("ScienceWorld adapter produced an empty task plan.")

    def reset_task(self, index: int | None = None) -> ResetResult:
        if self.env is None:
            raise RuntimeError("ScienceWorld adapter not set up.")

        task_index = 0 if index is None else index
        if task_index < 0 or task_index >= len(self.task_plan):
            raise IndexError(f"ScienceWorld task index out of range: {task_index}")

        task_name, variation_idx = self.task_plan[task_index]
        simplification = (self.runner_cfg.get("simplification") or "").strip()
        generate_gold_path = bool(self.runner_cfg.get("generate_gold_path", False))

        self.env.load(
            task_name,
            variationIdx=variation_idx,
            simplificationStr=simplification,
            generateGoldPath=generate_gold_path,
        )
        observation, info = self.env.reset()
        self.current_observation = observation
        self.current_valid_actions = self._extract_valid_actions(info)
        instruction = info.get("taskDesc") or self.env.get_task_description()
        self.current_task = TaskSpec(
            task_id=f"{task_name}[{variation_idx}]",
            instruction=instruction,
            task_type=task_name,
            task_description=instruction,
            metadata={
                "task_name": task_name,
                "variation_idx": variation_idx,
                "simplification": simplification,
            },
        )
        return ResetResult(
            task=self.current_task,
            observation=observation,
            valid_actions=self.current_valid_actions,
        )

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-6:]) if history_lines else "None"
        valid_actions = "\n".join(f"- {action}" for action in self.current_valid_actions[:80]) if self.current_valid_actions else "None"
        return (
            f"ScienceWorld task type: {task.task_type}\n"
            f"Goal:\n{task.instruction}\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Recent action history:\n{history_block}\n\n"
            f"Available actions:\n{valid_actions}\n\n"
            "Output exactly one action from the available actions list."
        )

    def step(self, action: str) -> StepResult:
        if self.env is None:
            raise RuntimeError("ScienceWorld step called before setup.")

        observation, reward, done, info = self.env.step(action)
        self.current_observation = observation
        self.current_valid_actions = self._extract_valid_actions(info)
        success = bool(done and info.get("score", 0) >= 100)
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(done),
            success=success,
            info=dict(info),
            valid_actions=self.current_valid_actions,
        )

    def force_finish(self) -> StepResult:
        info = {
            "forced_terminate": True,
            "valid": self.current_valid_actions,
        }
        return StepResult(
            observation=self.current_observation,
            reward=0.0,
            done=True,
            success=False,
            info=info,
            valid_actions=self.current_valid_actions,
        )

    def get_valid_actions(self) -> list[str]:
        return list(self.current_valid_actions)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _build_task_plan(self) -> list[tuple[str, int]]:
        if self.env is None:
            return []

        requested_task_names = self.runner_cfg.get("task_names") or []
        task_names = requested_task_names or self.env.get_task_names()
        variation_source = (self.runner_cfg.get("variation_source") or "train").strip().lower()
        explicit_variations = self.runner_cfg.get("variation_indices") or None

        if explicit_variations:
            candidate_variations = [int(item) for item in explicit_variations]
        elif variation_source == "dev":
            candidate_variations = [int(item) for item in self.env.get_variations_dev()]
        elif variation_source == "test":
            candidate_variations = [int(item) for item in self.env.get_variations_test()]
        else:
            candidate_variations = [int(item) for item in self.env.get_variations_train()]

        task_plan: list[tuple[str, int]] = []
        for task_name in task_names:
            max_variations = int(self.env.get_max_variations(task_name))
            for variation_idx in candidate_variations:
                if 0 <= variation_idx < max_variations:
                    task_plan.append((task_name, variation_idx))
        return task_plan

    @staticmethod
    def _extract_valid_actions(info: dict[str, Any]) -> list[str]:
        valid_actions = info.get("valid", []) if info else []
        return [str(action) for action in valid_actions]

