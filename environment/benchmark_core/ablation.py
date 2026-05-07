from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class AblationSpec:
    name: str
    memory_enabled: bool = True
    failure_memory_enabled: bool = True
    dopamine_rl_enabled: bool = True
    value_bias_enabled: bool = True
    hypergraph_enabled: bool = True
    consolidation_enabled: bool = True
    forgetting_enabled: bool = True
    delta_enabled: bool = True

    def diff_from(self, other: "AblationSpec") -> dict[str, tuple[object, object]]:
        diffs: dict[str, tuple[object, object]] = {}
        for item in fields(self):
            if item.name == "name":
                continue
            self_value = getattr(self, item.name)
            other_value = getattr(other, item.name)
            if self_value != other_value:
                diffs[item.name] = (self_value, other_value)
        return diffs


FULL = AblationSpec(name="full")

ABLATIONS = {
    "full": FULL,
    "no_memory": AblationSpec(
        name="no_memory",
        memory_enabled=False,
        failure_memory_enabled=False,
    ),
    "full_without_delta": AblationSpec(
        name="full_without_delta",
        delta_enabled=False,
    ),
    "full_with_delta": AblationSpec(
        name="full_with_delta",
    ),
    "no_FailureMemory": AblationSpec(
        name="no_FailureMemory",
        failure_memory_enabled=False,
    ),
    "no_DopamineRL": AblationSpec(
        name="no_DopamineRL",
        dopamine_rl_enabled=False,
    ),
    "no_ValueBias": AblationSpec(
        name="no_ValueBias",
        value_bias_enabled=False,
    ),
    "no_HyperGraph": AblationSpec(
        name="no_HyperGraph",
        hypergraph_enabled=False,
    ),
    "no_Consolidation": AblationSpec(
        name="no_Consolidation",
        consolidation_enabled=False,
    ),
    "no_Forgetting": AblationSpec(
        name="no_Forgetting",
        forgetting_enabled=False,
    ),
}


def get_ablation(name: str) -> AblationSpec:
    try:
        return ABLATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation: {name}") from exc


def ensure_supported(name: str, supported_flags: set[str] | None = None) -> AblationSpec:
    spec = get_ablation(name)
    if supported_flags is None:
        return spec

    diffs = spec.diff_from(FULL)
    unsupported = [flag for flag in diffs if flag not in supported_flags]
    if unsupported:
        unsupported_str = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            f"Ablation `{name}` changes unsupported brain mechanisms for the current backend: {unsupported_str}"
        )
    return spec
