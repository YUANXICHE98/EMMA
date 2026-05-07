from .ablation import ABLATIONS, AblationSpec, ensure_supported, get_ablation
from .adapter import BenchmarkAdapter, ResetResult, StepResult, TaskSpec


def __getattr__(name: str):
    if name == "MemRLBrain":
        from .brain import MemRLBrain

        return MemRLBrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
