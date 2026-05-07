from environment.benchmark_core import ABLATIONS, AblationSpec, BenchmarkAdapter, ResetResult, StepResult, TaskSpec, ensure_supported, get_ablation


def __getattr__(name: str):
    if name == "MemRLBrain":
        from environment.benchmark_core.brain import MemRLBrain

        return MemRLBrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
