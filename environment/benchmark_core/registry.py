from __future__ import annotations

from importlib import import_module


ADAPTERS = {
    "scienceworld": "scienceworld.adapter:ScienceWorldAdapter",
    "intercode_sql": "intercode_sql.adapter:InterCodeSQLAdapter",
    "bigcodebench": "bigcodebench.adapter:BigCodeBenchAdapter",
    "hle": "hle.adapter:HLEAdapter",
    "alfworld": "alfworld.adapter:ALFWorldAdapter",
    "lifelong_agent_bench": "lifelong_agent_bench.adapter:LifelongAgentBenchAdapter",
    "swebench": "swebench.adapter:SWEbenchAdapter",
    "webarena": "webarena.adapter:WebArenaAdapter",
    "osworld": "osworld.adapter:OSWorldAdapter",
}


def available_benchmarks() -> list[str]:
    return sorted(ADAPTERS.keys())


def make_adapter(benchmark_name: str, config: dict, traj_dir: str | None = None):
    try:
        target = ADAPTERS[benchmark_name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {benchmark_name}") from exc

    module_name, class_name = target.split(":")
    module = import_module(f"environment.{module_name}")
    cls = getattr(module, class_name)
    return cls(config=config, traj_dir=traj_dir)
