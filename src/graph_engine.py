"""Deterministic dependency-based failure propagation.

Model (see README "Simulation model"):

* Edge ``A -> B`` means "A depends on B". Failure flows from B to A (against the arrow).
* An edge is *broken* when its target has failed AND its declared fallback (if any) has not also failed.
* A component FAILS if any of its ``required`` edges is broken.
* A component is DEGRADED (but still operational) if only ``optional`` edges are broken.
* Propagation runs to a fixed point, so cycles terminate; the failed set only ever grows.

This is a reachability-style model. It has no load, capacity, timeout, retry or probability terms
and must not be read as a prediction of real production behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .models import Architecture, ArchitectureError


@dataclass
class FailureInfo:
    round: int  # 0 = initial failure, 1 = directly affected, 2+ = downstream cascade
    causes: list[str] = field(default_factory=list)  # broken required targets at the moment of failure


@dataclass
class SimulationResult:
    initial: list[str]
    failed: dict[str, FailureInfo]
    degraded: dict[str, list[str]]
    surviving: list[str]
    fallbacks_used: list[tuple[str, str, str]]  # (dependent, failed target, healthy fallback)
    waves: list[list[str]]

    @property
    def affected_ids(self) -> list[str]:
        return [c for wave in self.waves for c in wave]

    @property
    def affected_count(self) -> int:
        return len(self.failed)

    @property
    def direct(self) -> list[str]:
        return self.waves[1] if len(self.waves) > 1 else []

    @property
    def downstream(self) -> list[str]:
        return [c for wave in self.waves[2:] for c in wave]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_failure": list(self.initial),
            "affected_count": self.affected_count,
            "waves": [list(w) for w in self.waves],
            "failed": {k: {"round": v.round, "causes": v.causes} for k, v in self.failed.items()},
            "degraded": {k: list(v) for k, v in self.degraded.items()},
            "surviving": list(self.surviving),
            "fallbacks_used": [list(t) for t in self.fallbacks_used],
        }


def build_graph(arch: Architecture) -> nx.DiGraph:
    """Directed graph where edge A->B means 'A depends on B'."""
    g = nx.DiGraph()
    for c in arch.components:
        g.add_node(c.id, name=c.name, type=c.type)
    for d in arch.dependencies:
        g.add_edge(d.source, d.target, criticality=d.criticality, fallback=d.fallback,
                   provenance=d.provenance)
    return g


def simulate(arch: Architecture, failed_ids: str | list[str]) -> SimulationResult:
    initial = [failed_ids] if isinstance(failed_ids, str) else list(failed_ids)
    known = set(arch.component_ids())
    unknown = [c for c in initial if c not in known]
    if unknown:
        raise ArchitectureError(f"Unknown component id(s) for failure: {', '.join(unknown)}")
    if not initial:
        raise ArchitectureError("Choose at least one component to fail.")

    deps_by_source: dict[str, list] = {}
    for d in arch.dependencies:
        deps_by_source.setdefault(d.source, []).append(d)

    def broken(d, failed: dict[str, FailureInfo]) -> bool:
        return d.target in failed and not (d.fallback and d.fallback not in failed)

    failed: dict[str, FailureInfo] = {c: FailureInfo(0) for c in initial}
    waves: list[list[str]] = [sorted(set(initial), key=arch.component_ids().index)]
    rnd = 0
    while True:
        rnd += 1
        newly: dict[str, FailureInfo] = {}
        for comp in arch.components:  # fixed iteration order => reproducible
            if comp.id in failed:
                continue
            causes = [d.target for d in deps_by_source.get(comp.id, [])
                      if d.criticality == "required" and broken(d, failed)]
            if causes:
                newly[comp.id] = FailureInfo(rnd, causes)
        if not newly:
            break
        failed.update(newly)  # synchronous rounds: wave number = shortest failing path length
        waves.append(list(newly))

    degraded: dict[str, list[str]] = {}
    fallbacks_used: list[tuple[str, str, str]] = []
    for comp in arch.components:
        if comp.id in failed:
            continue
        for d in deps_by_source.get(comp.id, []):
            if d.target in failed:
                if broken(d, failed):  # only optional edges can still be broken here
                    degraded.setdefault(comp.id, []).append(d.target)
                else:
                    fallbacks_used.append((comp.id, d.target, d.fallback))
    surviving = [c.id for c in arch.components if c.id not in failed]
    return SimulationResult(initial, failed, degraded, surviving, fallbacks_used, waves)


def blast_radius(arch: Architecture) -> dict[str, int]:
    """Number of OTHER components affected if each component alone fails (same semantics as simulate)."""
    return {c.id: simulate(arch, c.id).affected_count - 1 for c in arch.components}


def shared_critical_dependencies(arch: Architecture, min_dependents: int = 2) -> list[dict[str, Any]]:
    """Components that >= min_dependents others directly require - the 'hidden single points of failure'."""
    out = []
    for c in arch.components:
        dependents = sorted(d.source for d in arch.dependencies
                            if d.target == c.id and d.criticality == "required" and not d.fallback)
        if len(dependents) >= min_dependents:
            out.append({"id": c.id, "name": c.name, "direct_dependents": dependents,
                        "blast_radius": simulate(arch, c.id).affected_count - 1})
    return sorted(out, key=lambda r: (-r["blast_radius"], r["id"]))
