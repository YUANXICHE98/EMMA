from __future__ import annotations

import json
import time
from typing import Any

from environment.memrl_core.adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


class InterCodeSQLAdapter(BenchmarkAdapter):
    benchmark_name = "intercode_sql"

    def __init__(self, config: dict[str, Any], traj_dir: str | None = None):
        super().__init__(config=config, traj_dir=traj_dir)
        self.env = None
        self.sql_image_name = None
        self.sql_test_data = None
        self.current_task = None

    def setup(self) -> None:
        from intercode.assets import sql_build_docker, sql_image_name, sql_test_data
        from intercode.envs import SqlEnv
        from intercode.envs.sql.sql_env import SQL_CONFIG
        from intercode.utils import get_container
        import mysql.connector

        self.sql_image_name = sql_image_name
        self.sql_test_data = sql_test_data

        if self.config.get("runner", {}).get("auto_build_docker", True):
            sql_build_docker()

        get_container(f"{self.sql_image_name}_ic_ctr", self.sql_image_name)
        self._wait_for_mysql_ready(mysql.connector, SQL_CONFIG)

        self.env = SqlEnv(
            self.sql_image_name,
            data_path=self.sql_test_data,
            traj_dir=self.traj_dir,
            verbose=False,
        )

    def reset_task(self, index: int | None = None) -> ResetResult:
        observation, _ = self.env.reset(index=index)
        extra = self.env.record.get("extra", {})
        db_name = extra.get("db", "unknown_db")
        self.env.exec_action(f"use `{db_name}`")
        if not self.env.info.get("action_executed", False):
            raise RuntimeError(f"Failed to switch InterCode-SQL session to database `{db_name}`: {self.env.observation}")
        self.current_task = TaskSpec(
            task_id=f"intercode_sql[{index if index is not None else 'random'}]",
            instruction=f"[db={db_name}] {self.env.query}",
            task_type="sql_generation",
            task_description=self.env.query,
            metadata={
                "db": db_name,
                "gold": self.env.gold,
                "query": self.env.query,
            },
        )
        return ResetResult(task=self.current_task, observation=self._stringify_observation(observation), valid_actions=[])

    def build_prompt(self, task: TaskSpec, observation: str, history_lines: list[str]) -> str:
        history_block = "\n".join(history_lines[-6:]) if history_lines else "None"
        db_name = task.metadata["db"]
        query = task.metadata["query"]
        last_action = self._extract_last_action(history_lines)
        last_status = self._current_status(observation)
        last_was_exploration = self._is_exploration_action(last_action)
        next_step_hint = ""
        if last_action:
            if last_status == "success" and not last_was_exploration:
                next_step_hint = (
                    "Immediate next-step policy:\n"
                    "- Your last non-exploration query executed successfully.\n"
                    "- If the current observation is the answer you intend to return, output `submit` now.\n"
                    "- Do not rewrite a successful answer query unless the observation clearly contradicts the user request.\n\n"
                )
            elif last_status == "error":
                next_step_hint = (
                    "Immediate next-step policy:\n"
                    "- Your last query failed.\n"
                    "- Inspect schema or repair the exact failing field/table/syntax issue before trying another final query.\n\n"
                )
        return (
            f"Goal: solve the SQL task against database `{db_name}`.\n"
            f"Natural-language query:\n{query}\n\n"
            "Rules:\n"
            "1. Output exactly one SQL statement, or output `submit` only if the current result already answers the task.\n"
            "2. The benchmark scores the final result rows after `submit`, not SQL surface form. Do not submit until the current observation already looks like the intended answer rows.\n"
            "3. Use short exploratory SQL first when schema is unclear, such as `show tables`, `describe <table>`, or `select * from <table> limit 5`.\n"
            "4. If you see `Unknown column`, `Unknown table`, or SQL syntax errors, inspect schema/value examples before trying another final query. Do not repeat the same broken query.\n"
            "5. If the query asks for names but you currently only have IDs, join back to the appropriate table to project the requested field.\n"
            "6. `select * from student` shows row values, not reliable column names. Use `describe student` before choosing name fields; do not guess columns like `firstname` or `last_name` unless you have seen them in schema output.\n"
            "7. Copy exact column names from prior `describe` output when writing the final query.\n"
            "8. If a complete answer query executes successfully with no SQL error, your next action should usually be `submit`. The benchmark scores the result of the last successful query, even if the result is `[]`.\n"
            "9. Never repeat an unchanged query that already executed successfully. If you believe that same query is your answer, output `submit` instead of repeating it.\n"
            "10. If two entity tables are connected through a bridge table discovered in schema (for example a table with keys like `StuID` and `PetID`), join through that bridge table. Do not assume one entity table directly contains the other's ID.\n"
            "11. Return a complete SQL statement on one line. Do not explain your reasoning.\n\n"
            f"Current execution status: {last_status}\n"
            f"Last action type: {'exploration' if last_was_exploration else 'candidate_answer'}\n\n"
            f"{next_step_hint}"
            f"Current observation/result:\n{observation}\n\n"
            f"Recent action history:\n{history_block}"
        )

    def step(self, action: str) -> StepResult:
        observation, reward, done, info = self.env.step(action)
        success = bool(done and reward >= self.config.get("runner", {}).get("submit_reward_threshold", 0.99))
        return StepResult(
            observation=self._stringify_observation(observation),
            reward=float(reward),
            done=bool(done),
            success=success,
            info=dict(info),
            valid_actions=[],
        )

    def force_finish(self) -> StepResult:
        observation, reward, done, info = self.env.step("submit")
        success = bool(reward >= self.config.get("runner", {}).get("submit_reward_threshold", 0.99))
        return StepResult(
            observation=self._stringify_observation(observation),
            reward=float(reward),
            done=bool(done),
            success=success,
            info=dict(info),
            valid_actions=["submit"],
        )

    def normalize_action(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if not action:
            return "submit"
        if action.lower().startswith("```"):
            lines = [line for line in action.splitlines() if not line.strip().startswith("```")]
            action = "\n".join(lines).strip()
        if action.lower().startswith("sql\n"):
            action = action[4:].strip()
        if action.lower().startswith("action:"):
            action = action.split(":", 1)[1].strip()
        return action.strip()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    @staticmethod
    def _wait_for_mysql_ready(mysql_connector: Any, sql_config: dict[str, Any], timeout_seconds: int = 60) -> None:
        deadline = time.time() + timeout_seconds
        last_error = None
        while time.time() < deadline:
            conn = None
            try:
                conn = mysql_connector.connect(**sql_config)
                conn.close()
                return
            except Exception as exc:  # pragma: no cover - readiness varies by local Docker startup
                last_error = exc
                time.sleep(2)
            finally:
                if conn is not None and conn.is_connected():
                    conn.close()
        raise RuntimeError(f"MySQL container did not become ready within {timeout_seconds}s: {last_error}")

    @staticmethod
    def _stringify_observation(observation: Any) -> str:
        if observation is None:
            return "None"
        if isinstance(observation, list):
            return json.dumps(observation[:8], ensure_ascii=False)
        return str(observation)

    @staticmethod
    def _extract_last_action(history_lines: list[str]) -> str:
        if not history_lines:
            return ""
        last_item = history_lines[-1]
        if last_item.startswith("> "):
            return last_item[2:].splitlines()[0].strip()
        return ""

    @staticmethod
    def _current_status(observation: str) -> str:
        text = (observation or "").strip().lower()
        if text.startswith("error executing query:"):
            return "error"
        if text in {"none", ""}:
            return "unknown"
        return "success"

    @staticmethod
    def _is_exploration_action(action: str) -> bool:
        normalized = (action or "").strip().lower().rstrip(";")
        if not normalized:
            return True
        if normalized in {"show tables", "submit"}:
            return True
        if normalized.startswith("describe "):
            return True
        if normalized.startswith("select * from "):
            return True
        return False
