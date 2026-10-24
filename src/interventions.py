"""What-if interventions and baseline comparison.

Interventions modify a *copy* of the architecture; the simulator then runs unchanged on it, so the
baseline and the intervention scenario share identical propagation semantics.

Kinds:
* ``isolate``          - remove the dependency edge source->target for the scenario (e.g. make the call
                         asynchronous / drop the coupling). Always applicable if the edge exists.
* ``circuit_breaker``  - open breaker on source->target: the call fails fast and ``source`` runs degraded
                         instead of failing. Applicable ONLY if ``source`` declares ``supports_degraded_mode``.
* ``fallback``         - declare ``fallback`` component as a substitute for target on source->target.
                         It only helps if the fallback component itself has not failed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .graph_engine import SimulationResult, simulate
from .models import Architecture, ArchitectureError

KINDS = ("isolate", "circuit_breaker", "fallback")


@dataclass
class Intervention:
    kind: str
    source: str
    target: str
    fallback: str | None = None

    def describe(self, arch: Architecture) -> str:
        s, t = arch.name_of(self.source), arch.name_of(self.target)
        if self.kind == "isolate":
            return f"Isolate: remove dependency {s} -> {t}"
        if self.kind == "circuit_breaker":
            return f"Circuit breaker on {s} -> {t} ({s} runs degraded instead of failing)"
        return f"Fallback: {s} uses {arch.name_of(self.fallback or '')} when {t} fails"


def apply_interventions(arch: Architecture, interventions: list[Intervention]) -> Architecture:
    new = arch.copy()
    for iv in interventions:
        if iv.kind not in KINDS:
            raise ArchitectureError(f"Unknown intervention kind '{iv.kind}'.")
        dep = next((d for d in new.dependencies if d.key == (iv.source, iv.target)), None)
        if dep is None:
            raise ArchitectureError(f"No dependency {iv.source} -> {iv.target} exists to modify.")
        if iv.kind == "isolate":
            new.dependencies.remove(dep)
        elif iv.kind == "circuit_breaker":
            comp = new.get(iv.source)
            if not comp.supports_degraded_mode:
                raise ArchitectureError(
                    f"Circuit breaker not applicable: '{comp.name}' does not declare a degraded mode, "
                    "so it cannot keep serving when this dependency is open.")
            dep.criticality = "optional"
        else:
            if not iv.fallback or iv.fallback not in new.component_ids():
                raise ArchitectureError(f"Fallback component '{iv.fallback}' does not exist.")
            if iv.fallback in (iv.source, iv.target):
                raise ArchitectureError("Fallback must be a different component from source and target.")
            dep.fallback = iv.fallback
    return new


@dataclass
class Comparison:
    failed_component: str
    interventions: list[Intervention]
    baseline: SimulationResult
    scenario: SimulationResult
    protected: list[str]
    still_affected: list[str]
    newly_affected: list[str]
    reduction: int

    @property
    def helps(self) -> bool:
        return self.reduction > 0

    def verdict(self) -> str:
        if self.reduction > 0:
            return f"Reduces affected components by {self.reduction} ({self.baseline.affected_count} -> {self.scenario.affected_count})."
        if self.reduction == 0:
            return "No reduction: these interventions do not change the affected set for this failure under the model."
        return "Increases affected components (unexpected for these interventions)."

    def to_dict(self, arch: Architecture) -> dict[str, Any]:
        return {
            "failed_component": self.failed_component,
            "interventions": [{**asdict(i), "description": i.describe(arch)} for i in self.interventions],
            "baseline": self.baseline.to_dict(),
            "with_interventions": self.scenario.to_dict(),
            "protected": self.protected,
            "still_affected": self.still_affected,
            "newly_affected": self.newly_affected,
            "absolute_reduction": self.reduction,
            "verdict": self.verdict(),
        }


def compare(arch: Architecture, failed_component: str, interventions: list[Intervention]) -> Comparison:
    baseline = simulate(arch, failed_component)
    scenario = simulate(apply_interventions(arch, interventions), failed_component)
    b, s = set(baseline.failed), set(scenario.failed)
    order = arch.component_ids()
    key = order.index
    return Comparison(
        failed_component, list(interventions), baseline, scenario,
        protected=sorted(b - s, key=key), still_affected=sorted(b & s, key=key),
        newly_affected=sorted(s - b, key=key), reduction=len(b) - len(s),
    )
