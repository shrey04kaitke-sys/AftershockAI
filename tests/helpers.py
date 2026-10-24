from src.models import Architecture, Component, Dependency
from src.system_parser import validate_architecture


def make_arch(ids, edges, degraded=(), name="test"):
    """edges: (source, target[, criticality[, fallback]]) -- source depends on target."""
    comps = [Component(id=i, name=i.upper(), supports_degraded_mode=i in degraded, provenance="sample_data") for i in ids]
    deps = []
    for e in edges:
        crit = e[2] if len(e) > 2 else "required"
        fb = e[3] if len(e) > 3 else None
        deps.append(Dependency(e[0], e[1], crit, fb, "sample_data"))
    return validate_architecture(Architecture(comps, deps, name=name))
