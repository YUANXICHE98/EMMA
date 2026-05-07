from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ENV_ROOT = Path(__file__).resolve().parent
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))
if str(ENV_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT.parent))

from memrl_core.ablation import ABLATIONS
from memrl_core.brain import MemRLBrain
from memrl_core.registry import available_benchmarks, make_adapter

WORKSPACE_ROOT = ENV_ROOT.parent
EXP_ROOT = WORKSPACE_ROOT / "exp"


def _configure_local_proxy() -> None:
    """
    Allow the EMMA startup path to route outbound HTTP(S) traffic through
    an explicitly enabled local proxy. By default, leave the ambient
    environment untouched.
    """
    proxy_enabled = str(os.environ.get("MEMRL_ENABLE_LOCAL_PROXY", "")).strip().lower()
    if proxy_enabled not in {"1", "true", "yes", "on"}:
        return

    proxy_url = os.environ.get("MEMRL_PROXY_URL", "").strip()
    proxy_port = os.environ.get("MEMRL_LOCAL_PROXY_PORT", "").strip()

    if not proxy_url and proxy_port:
        proxy_url = f"http://127.0.0.1:{proxy_port}"

    if not proxy_url:
        return

    no_proxy = os.environ.get("NO_PROXY", "").strip()
    localhost_bypass = "127.0.0.1,localhost"

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[key] = proxy_url

    if no_proxy:
        if localhost_bypass not in no_proxy:
            os.environ["NO_PROXY"] = f"{no_proxy},{localhost_bypass}"
            os.environ["no_proxy"] = os.environ["NO_PROXY"]
    else:
        os.environ["NO_PROXY"] = localhost_bypass
        os.environ["no_proxy"] = localhost_bypass


def _getenv(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_args(default_benchmark: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared EMMA brain on a benchmark adapter.")
    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        choices=available_benchmarks(),
        required=default_benchmark is None,
    )
    parser.add_argument("--condition", default="full", choices=sorted(ABLATIONS.keys()))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--traj-dir", type=Path, default=None)
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_benchmark_config(benchmark_name: str) -> dict[str, Any]:
    _configure_local_proxy()
    benchmark_config_path = ENV_ROOT / benchmark_name / "config.yaml"
    with benchmark_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    legacy_config_path = EXP_ROOT / "configs" / "memrl_config.yaml"
    legacy = {}
    if legacy_config_path.exists():
        with legacy_config_path.open("r", encoding="utf-8") as handle:
            legacy = yaml.safe_load(handle) or {}
        for key in ("llm", "encoder", "memory", "rl"):
            config.setdefault(key, {})
            for legacy_key, legacy_value in legacy.get(key, {}).items():
                config[key].setdefault(legacy_key, legacy_value)

    if not os.environ.get("OPENAI_API_KEY"):
        legacy_api_key = legacy.get("llm", {}).get("api_key")
        if legacy_api_key:
            config.setdefault("llm", {})
            config["llm"].setdefault("api_key", legacy_api_key)

    config.setdefault("llm", {})
    config.setdefault("encoder", {})
    config.setdefault("runner", {})
    config.setdefault("routing", {})
    env_api_key = _getenv("EMMA_OPENAI_API_KEY", "MEMRL_OPENAI_API_KEY", "OPENAI_API_KEY")
    env_base_url = _getenv("EMMA_OPENAI_BASE_URL", "MEMRL_OPENAI_BASE_URL", "OPENAI_BASE_URL")
    env_model_name = _getenv("EMMA_OPENAI_MODEL", "EMMA_LLM_MODEL", "MEMRL_OPENAI_MODEL", "MEMRL_LLM_MODEL", "OPENAI_MODEL")
    env_protocol = _getenv("EMMA_OPENAI_PROTOCOL", "EMMA_LLM_PROTOCOL", "MEMRL_OPENAI_PROTOCOL", "MEMRL_LLM_PROTOCOL")
    env_action_max_tokens = _getenv("EMMA_OPENAI_ACTION_MAX_TOKENS", "EMMA_ACTION_MAX_TOKENS", "MEMRL_OPENAI_ACTION_MAX_TOKENS", "MEMRL_ACTION_MAX_TOKENS")
    env_action_timeout = _getenv("EMMA_OPENAI_ACTION_TIMEOUT", "EMMA_ACTION_TIMEOUT", "MEMRL_OPENAI_ACTION_TIMEOUT", "MEMRL_ACTION_TIMEOUT")
    env_responses_reasoning_effort = _getenv("EMMA_OPENAI_REASONING_EFFORT", "EMMA_RESPONSES_REASONING_EFFORT", "MEMRL_OPENAI_REASONING_EFFORT", "MEMRL_RESPONSES_REASONING_EFFORT")
    env_responses_text_verbosity = _getenv("EMMA_OPENAI_TEXT_VERBOSITY", "EMMA_RESPONSES_TEXT_VERBOSITY", "MEMRL_OPENAI_TEXT_VERBOSITY", "MEMRL_RESPONSES_TEXT_VERBOSITY")
    embedding_api_key = _getenv("EMMA_EMBEDDING_API_KEY", "MEMRL_EMBEDDING_API_KEY", "EMBEDDING_API_KEY")
    embedding_base_url = _getenv("EMMA_EMBEDDING_BASE_URL", "MEMRL_EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL")
    embedding_model_name = _getenv("EMMA_EMBEDDING_MODEL", "MEMRL_EMBEDDING_MODEL", "EMBEDDING_MODEL")
    judge_mode = _getenv("EMMA_HLE_JUDGE_MODE", "MEMRL_HLE_JUDGE_MODE", "HLE_JUDGE_MODE")
    judge_model = _getenv("EMMA_HLE_JUDGE_MODEL", "MEMRL_HLE_JUDGE_MODEL", "HLE_JUDGE_MODEL")
    routing_enabled = _getenv("EMMA_ROUTING_ENABLED", "MEMRL_ROUTING_ENABLED")
    routing_primary_model = _getenv("EMMA_ROUTING_PRIMARY_MODEL", "MEMRL_ROUTING_PRIMARY_MODEL")
    routing_secondary_model = _getenv("EMMA_ROUTING_SECONDARY_MODEL", "MEMRL_ROUTING_SECONDARY_MODEL")
    routing_secondary_protocol = _getenv("EMMA_ROUTING_SECONDARY_PROTOCOL", "MEMRL_ROUTING_SECONDARY_PROTOCOL")
    routing_trigger_mode = _getenv("EMMA_ROUTING_TRIGGER_MODE", "MEMRL_ROUTING_TRIGGER_MODE")
    if env_api_key:
        config["llm"]["api_key"] = env_api_key
    if env_base_url:
        config["llm"]["base_url"] = env_base_url
    if env_model_name:
        config["llm"]["model_name"] = env_model_name
    if env_protocol:
        config["llm"]["protocol"] = env_protocol
    if env_action_max_tokens:
        config["llm"]["action_max_tokens"] = int(env_action_max_tokens)
    if env_action_timeout:
        config["llm"]["action_timeout"] = float(env_action_timeout)
    if env_responses_reasoning_effort:
        config["llm"]["responses_reasoning_effort"] = env_responses_reasoning_effort
    if env_responses_text_verbosity:
        config["llm"]["responses_text_verbosity"] = env_responses_text_verbosity
    if embedding_api_key:
        config["encoder"]["api_key"] = embedding_api_key
    if embedding_base_url:
        config["encoder"]["base_url"] = embedding_base_url
    if embedding_model_name:
        config["encoder"]["model_name"] = embedding_model_name
    if judge_mode:
        config["runner"]["judge_mode"] = judge_mode
    if judge_model:
        config["runner"]["judge_model"] = judge_model
    if routing_enabled is not None:
        config["routing"]["enabled"] = str(routing_enabled).strip().lower() in {"1", "true", "yes", "on"}
    if routing_primary_model:
        config["routing"]["primary_model"] = routing_primary_model
        config["llm"]["model_name"] = routing_primary_model
    if routing_secondary_model:
        config["routing"]["secondary_model"] = routing_secondary_model
    if routing_secondary_protocol:
        config["routing"]["secondary_protocol"] = routing_secondary_protocol
    if routing_trigger_mode:
        config["routing"]["trigger_mode"] = routing_trigger_mode
        config["runner"]["routing_trigger_mode"] = routing_trigger_mode

    return config


def main(default_benchmark: str | None = None) -> None:
    args = parse_args(default_benchmark=default_benchmark)
    config = load_benchmark_config(args.benchmark)
    runner_cfg = config.get("runner", {})

    results_dir = args.results_dir or (ENV_ROOT / args.benchmark / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = args.traj_dir or results_dir / "trajectories" / args.condition
    traj_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.results_file or results_dir / f"{args.benchmark}_{args.condition}.json"

    episodes = args.episodes or runner_cfg.get("default_num_episodes", 3)
    max_steps = args.max_steps or runner_cfg.get("max_steps_per_episode", 8)

    adapter = make_adapter(args.benchmark, config=config, traj_dir=str(traj_dir))
    adapter.setup()

    brain = MemRLBrain(config=config, results_dir=results_dir, condition=args.condition)
    ablation = brain.activate_ablation(args.condition)

    results = []
    try:
        for offset in range(episodes):
            task_index = args.start_index + offset
            print(f"[episode] benchmark={args.benchmark} condition={args.condition} task_index={task_index}")
            episode_result = brain.run_episode(
                adapter=adapter,
                task_index=task_index,
                max_steps=max_steps,
                ablation=ablation,
            )
            results.append(episode_result)
            print(
                f"[result] success={episode_result['success']} reward={episode_result['reward']:.2f} "
                f"steps={episode_result['steps']} task_id={episode_result['task_id']}"
            )
    finally:
        adapter.close()

    brain.save_summary(
        benchmark_name=args.benchmark,
        ablation=ablation,
        start_index=args.start_index,
        results=results,
        results_file=results_file,
    )
    print(f"[saved] {results_file}")


if __name__ == "__main__":
    main()
