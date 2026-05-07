from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class SWEbenchAdapter(BenchmarkAdapter):
    benchmark_name = "swebench"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.runner_cfg = config.get("runner", {})
        self.instances: list[dict[str, Any]] = []
        self.current_instance: dict[str, Any] | None = None
        self.current_task: TaskSpec | None = None
        self.current_observation = ""
        self.runtime_dir = Path(traj_dir) if traj_dir else Path("environment/swebench/runtime")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        self.instances = self._load_instances()
        if not self.instances:
            raise RuntimeError("SWE-bench adapter loaded zero instances.")

    def reset_task(self, index: int | None = None) -> ResetResult:
        if not self.instances:
            raise RuntimeError("SWE-bench adapter has no loaded instances.")

        task_index = 0 if index is None else index
        if task_index < 0 or task_index >= len(self.instances):
            raise IndexError(f"SWE-bench instance index out of range: {task_index}")

        instance = self.instances[task_index]
        self.current_instance = instance
        observation = self._build_observation(instance)
        self.current_observation = observation
        self.current_task = TaskSpec(
            task_id=instance["instance_id"],
            instruction=instance["problem_statement"],
            task_type="swebench_patch",
            task_description=instance["problem_statement"],
            metadata={
                "repo": instance.get("repo", ""),
                "base_commit": instance.get("base_commit", ""),
                "dataset_name": self.runner_cfg.get("dataset_name", "SWE-bench/SWE-bench_Lite"),
                "split": self.runner_cfg.get("split", "test"),
                "hints_text": instance.get("hints_text", ""),
                "instance_id": instance["instance_id"],
            },
        )
        return ResetResult(task=self.current_task, observation=observation, valid_actions=[])

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-4:]) if history_lines else "None"
        hints_text = task.metadata.get("hints_text", "").strip()
        hints_block = f"\n\nHints:\n{hints_text}" if hints_text else ""
        return (
            f"SWE-bench issue for repository `{task.metadata.get('repo', '')}`.\n"
            f"Base commit: {task.metadata.get('base_commit', '')}\n\n"
            f"Issue statement:\n{task.task_description or task.instruction}"
            f"{hints_block}\n\n"
            f"Current benchmark context:\n{observation}\n\n"
            f"Recent patch attempts:\n{history_block}\n\n"
            "Output exactly one unified diff patch that resolves the issue. "
            "Do not include commentary, markdown fences, or explanations."
        )

    def step(self, action: str) -> StepResult:
        if self.current_instance is None:
            raise RuntimeError("SWE-bench step called before reset_task.")

        model_patch = self.normalize_action(action)
        eval_result = self._run_official_harness(self.current_instance, model_patch)
        resolved = bool(eval_result["resolved"])
        observation = self._build_result_observation(eval_result)
        self.current_observation = observation
        return StepResult(
            observation=observation,
            reward=1.0 if resolved else 0.0,
            done=True,
            success=resolved,
            info=eval_result,
            valid_actions=[],
        )

    def force_finish(self) -> StepResult:
        return StepResult(
            observation=self.current_observation or "SWE-bench episode terminated without a submitted patch.",
            reward=0.0,
            done=True,
            success=False,
            info={"forced_terminate": True},
            valid_actions=[],
        )

    def normalize_action(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if action.lower().startswith("```"):
            lines = [line for line in action.splitlines() if not line.strip().startswith("```")]
            action = "\n".join(lines).strip()
        if action.lower().startswith("diff\n"):
            action = action[5:].strip()
        if action.lower().startswith("patch:\n"):
            action = action[7:].strip()
        return action

    def _load_instances(self) -> list[dict[str, Any]]:
        task_file = (self.runner_cfg.get("task_file") or "").strip()
        if task_file:
            return self._load_instances_from_file(Path(task_file))

        dataset_name = self.runner_cfg.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = self.runner_cfg.get("split", "test")
        instance_ids = self.runner_cfg.get("instance_ids") or None

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "SWE-bench adapter requires either runner.task_file or the `datasets` package "
                f"to load {dataset_name}:{split}."
            ) from exc

        dataset = load_dataset(dataset_name, split=split)
        instances: list[dict[str, Any]] = []
        allowed = set(instance_ids or [])
        for record in dataset:
            if allowed and record.get("instance_id") not in allowed:
                continue
            instances.append(dict(record))
        return instances

    @staticmethod
    def _load_instances_from_file(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"SWE-bench task_file not found: {path}")
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [dict(item) for item in payload]
            raise ValueError(f"SWE-bench JSON task_file must contain a list: {path}")

        instances = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
        return instances

    def _run_official_harness(self, instance: dict[str, Any], model_patch: str) -> dict[str, Any]:
        python_bin = self.runner_cfg.get("python_bin") or sys.executable
        dataset_name = self.runner_cfg.get("dataset_name", "SWE-bench/SWE-bench_Lite")
        split = self.runner_cfg.get("split", "test")
        max_workers = str(self.runner_cfg.get("max_workers", 1))
        timeout = str(self.runner_cfg.get("timeout", 1800))
        run_id = f"{self.runner_cfg.get('run_id_prefix', 'memrl_swebench')}_{instance['instance_id']}_{uuid4().hex[:8]}"

        predictions_dir = self.runtime_dir / run_id
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "predictions.jsonl"
        prediction = {
            "instance_id": instance["instance_id"],
            "model_name_or_path": f"memrl/{self.benchmark_name}",
            "model_patch": model_patch,
        }
        predictions_path.write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")

        cmd = [
            python_bin,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--instance_ids",
            instance["instance_id"],
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            max_workers,
            "--run_id",
            run_id,
            "--timeout",
            timeout,
        ]

        namespace = self.runner_cfg.get("namespace", "")
        cmd.extend(["--namespace", namespace])
        if self.runner_cfg.get("force_rebuild", False):
            cmd.extend(["--force_rebuild", "true"])

        completed = subprocess.run(
            cmd,
            cwd=str(predictions_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        report_path = (
            predictions_dir
            / "logs"
            / "run_evaluation"
            / prediction["model_name_or_path"].replace("/", "__")
            / instance["instance_id"]
            / "report.json"
        )
        resolved = False
        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            resolved = bool(report.get(instance["instance_id"], {}).get("resolved", False))

        return {
            "run_id": run_id,
            "instance_id": instance["instance_id"],
            "resolved": resolved,
            "report_path": str(report_path),
            "report": report,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "returncode": completed.returncode,
            "predictions_path": str(predictions_path),
        }

    @staticmethod
    def _build_observation(instance: dict[str, Any]) -> str:
        repo = instance.get("repo", "")
        base_commit = instance.get("base_commit", "")
        hints = (instance.get("hints_text") or "").strip()
        hints_block = f"\nHints:\n{hints}" if hints else ""
        return (
            f"Repository: {repo}\n"
            f"Base commit: {base_commit}\n"
            f"Instance: {instance.get('instance_id', '')}\n"
            f"Issue:\n{instance.get('problem_statement', '')}"
            f"{hints_block}"
        )

    @staticmethod
    def _build_result_observation(eval_result: dict[str, Any]) -> str:
        status = "resolved" if eval_result.get("resolved") else "unresolved"
        return (
            f"SWE-bench evaluation finished.\n"
            f"Status: {status}\n"
            f"Return code: {eval_result.get('returncode')}\n"
            f"Report path: {eval_result.get('report_path')}\n\n"
            f"stdout tail:\n{eval_result.get('stdout', '')[-4000:]}\n\n"
            f"stderr tail:\n{eval_result.get('stderr', '')[-4000:]}"
        )
