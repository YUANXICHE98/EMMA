from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from environment.run_memrl_benchmark import ENV_ROOT, load_benchmark_config
from environment.memrl_core.ablation import ABLATIONS
from environment.memrl_core.brain import MemRLBrain
from environment.memrl_core.registry import make_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BigCodeBench with the shared MemRL brain under a multi-epoch train/val protocol."
    )
    parser.add_argument("--condition", default="full", choices=sorted(ABLATIONS.keys()))
    parser.add_argument(
        "--protocol",
        default=None,
        choices=["memrl_style_train_val", "emma_online_evolving"],
        help="Experiment protocol: MemRL-style read-only validation or EMMA-style continuous online evolution.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Total selected tasks across train and val.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--run-validation", dest="run_validation", action="store_true")
    parser.add_argument("--no-validation", dest="run_validation", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--traj-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.set_defaults(run_validation=None)
    return parser.parse_args()


def _load_split_from_file(split_file: Path, valid_task_ids: set[str]) -> tuple[list[str], list[str]]:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = [str(task_id) for task_id in payload.get("train_ids", []) if str(task_id) in valid_task_ids]
    val_ids = [str(task_id) for task_id in payload.get("val_ids", []) if str(task_id) in valid_task_ids]
    return train_ids, val_ids


def _split_task_indices(
    task_records: list[dict[str, Any]],
    *,
    start_index: int,
    total_tasks: int,
    train_ratio: float,
    split_seed: int,
    split_file: Path | None,
) -> tuple[list[int], list[int]]:
    if split_file is not None and split_file.exists():
        task_id_to_index = {str(record["task_id"]): idx for idx, record in enumerate(task_records)}
        train_ids, val_ids = _load_split_from_file(split_file, set(task_id_to_index))
        train_indices = [task_id_to_index[task_id] for task_id in train_ids if task_id in task_id_to_index]
        val_indices = [task_id_to_index[task_id] for task_id in val_ids if task_id in task_id_to_index]
        return train_indices, val_indices

    if total_tasks <= 0:
        return [], []

    stop_index = min(len(task_records), start_index + total_tasks)
    selected_indices = list(range(start_index, stop_index))
    selected_records = [task_records[idx] for idx in selected_indices]
    task_id_to_index = {str(record["task_id"]): idx for idx, record in zip(selected_indices, selected_records)}

    shuffled = list(selected_indices)
    random.Random(split_seed).shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    split_point = max(1, split_point) if len(shuffled) > 1 else len(shuffled)
    split_point = min(split_point, len(shuffled))
    train_indices = shuffled[:split_point]
    val_indices = shuffled[split_point:]
    return train_indices, val_indices


def _phase_results_file(epoch_dir: Path, phase: str, condition: str) -> Path:
    return epoch_dir / phase / f"bigcodebench_{condition}.json"


def _save_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_name = "bigcodebench"
    config = load_benchmark_config(benchmark_name)
    runner_cfg = config.get("runner", {})

    results_dir = args.results_dir or (ENV_ROOT / benchmark_name / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = args.traj_dir or results_dir / "trajectories" / args.condition
    traj_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = args.episodes or int(runner_cfg.get("default_num_episodes", 5))
    epochs = args.epochs or int(runner_cfg.get("num_epochs", 3))
    protocol = str(args.protocol or runner_cfg.get("protocol", "memrl_style_train_val")).strip().lower()
    train_ratio = float(args.train_ratio if args.train_ratio is not None else runner_cfg.get("train_ratio", 0.7))
    split_seed = int(args.split_seed if args.split_seed is not None else runner_cfg.get("split_seed", 42))
    max_steps = args.max_steps or int(runner_cfg.get("max_steps_per_episode", 1))
    if args.run_validation is None:
        run_validation = bool(runner_cfg.get("run_validation", True))
    else:
        run_validation = bool(args.run_validation)

    adapter = make_adapter(benchmark_name, config=config, traj_dir=str(traj_dir))
    adapter.setup()

    train_indices, val_indices = _split_task_indices(
        adapter.task_records,
        start_index=args.start_index,
        total_tasks=total_tasks,
        train_ratio=train_ratio,
        split_seed=split_seed,
        split_file=args.split_file,
    )

    brain = MemRLBrain(config=config, results_dir=results_dir, condition=args.condition)
    ablation = brain.activate_ablation(args.condition)
    brain.initialize_seed_memory(adapter, ablation)

    run_config = {
        "benchmark": benchmark_name,
        "condition": args.condition,
        "protocol": protocol,
        "total_selected_tasks": len(train_indices) + len(val_indices),
        "start_index": args.start_index,
        "epochs": epochs,
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "run_validation": run_validation,
        "train_task_indices": train_indices,
        "val_task_indices": val_indices,
        "memory_file": str(results_dir / f"memrl_memory_{args.condition}.json"),
    }
    _save_run_config(results_dir / "run_config.json", run_config)

    epoch_summaries: list[dict[str, Any]] = []
    try:
        for epoch in range(1, epochs + 1):
            epoch_dir = results_dir / f"epoch{epoch}"
            (epoch_dir / "train").mkdir(parents=True, exist_ok=True)
            if run_validation:
                (epoch_dir / "val").mkdir(parents=True, exist_ok=True)

            train_results = []
            for task_index in train_indices:
                print(
                    f"[episode] benchmark={benchmark_name} condition={args.condition} epoch={epoch} "
                    f"phase=train task_index={task_index}"
                )
                episode_result = brain.run_episode(
                    adapter=adapter,
                    task_index=task_index,
                    max_steps=max_steps,
                    ablation=ablation,
                    allow_retrieval=True,
                    allow_value_update=True,
                    allow_memory_write=True,
                )
                train_results.append(episode_result)
                print(
                    f"[result] epoch={epoch} phase=train success={episode_result['success']} "
                    f"reward={episode_result['reward']:.2f} task_id={episode_result['task_id']}"
                )

            train_results_file = _phase_results_file(epoch_dir, "train", args.condition)
            brain.save_summary(
                benchmark_name=benchmark_name,
                ablation=ablation,
                start_index=args.start_index,
                results=train_results,
                results_file=train_results_file,
            )

            val_results = []
            if run_validation and val_indices:
                val_allow_value_update = protocol == "emma_online_evolving"
                val_allow_memory_write = protocol == "emma_online_evolving"
                for task_index in val_indices:
                    print(
                        f"[episode] benchmark={benchmark_name} condition={args.condition} epoch={epoch} "
                        f"phase=val task_index={task_index}"
                    )
                    episode_result = brain.run_episode(
                        adapter=adapter,
                        task_index=task_index,
                        max_steps=max_steps,
                        ablation=ablation,
                        allow_retrieval=True,
                        allow_value_update=val_allow_value_update,
                        allow_memory_write=val_allow_memory_write,
                    )
                    val_results.append(episode_result)
                    print(
                        f"[result] epoch={epoch} phase=val success={episode_result['success']} "
                        f"reward={episode_result['reward']:.2f} task_id={episode_result['task_id']}"
                    )

                val_results_file = _phase_results_file(epoch_dir, "val", args.condition)
                brain.save_summary(
                    benchmark_name=benchmark_name,
                    ablation=ablation,
                    start_index=args.start_index,
                    results=val_results,
                    results_file=val_results_file,
                )

            epoch_summary = {
                "epoch": epoch,
                "train_results_file": str(train_results_file),
                "val_results_file": str(_phase_results_file(epoch_dir, "val", args.condition)) if val_results else "",
                "train_episodes": len(train_results),
                "train_success_rate": round(
                    sum(1 for result in train_results if result["success"]) / len(train_results), 4
                )
                if train_results
                else 0.0,
                "val_episodes": len(val_results),
                "val_success_rate": round(sum(1 for result in val_results if result["success"]) / len(val_results), 4)
                if val_results
                else 0.0,
            }
            _save_run_config(epoch_dir / "epoch_summary.json", epoch_summary)
            epoch_summaries.append(epoch_summary)
    finally:
        adapter.close()

    _save_run_config(
        results_dir / "summary.json",
        {
            "benchmark": benchmark_name,
            "condition": args.condition,
            "protocol": protocol,
            "epochs": epoch_summaries,
            "memory_file": str(results_dir / f"memrl_memory_{args.condition}.json"),
        },
    )
    print(f"[saved] {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
