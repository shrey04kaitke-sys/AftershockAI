import pytest

from src.graph_engine import blast_radius, shared_critical_dependencies, simulate
from src.models import ArchitectureError
from src.samples import load_sample_architecture
from tests.helpers import make_arch


def test_single_component_failure_with_no_dependents():
    arch = make_arch(["a", "b"], [])
    r = simulate(arch, "a")
    assert set(r.failed) == {"a"} and r.surviving == ["b"] and r.affected_count == 1


def test_linear_chain_direction_and_waves():
    # a depends on b, b depends on c
    arch = make_arch(["a", "b", "c"], [("a", "b"), ("b", "c")])
    r = simulate(arch, "c")
    assert r.waves == [["c"], ["b"], ["a"]]
    assert r.direct == ["b"] and r.downstream == ["a"]
    assert r.failed["a"].causes == ["b"]


def test_dependency_direction_is_enforced():
    # a depends on b: failing the DEPENDENT must not take down its dependency
    arch = make_arch(["a", "b"], [("a", "b")])
    assert set(simulate(arch, "a").failed) == {"a"}
    assert set(simulate(arch, "b").failed) == {"a", "b"}


def test_branching_graph_and_surviving_independent():
    arch = make_arch(["gw", "x", "y", "db", "solo"], [("gw", "x"), ("gw", "y"), ("x", "db"), ("y", "db")])
    r = simulate(arch, "db")
    assert set(r.failed) == {"db", "x", "y", "gw"}
    assert r.surviving == ["solo"]


def test_optional_dependency_only_degrades():
    arch = make_arch(["svc", "cache"], [("svc", "cache", "optional")])
    r = simulate(arch, "cache")
    assert set(r.failed) == {"cache"} and r.degraded == {"svc": ["cache"]} and "svc" in r.surviving


def test_cycle_terminates_and_propagates():
    arch = make_arch(["a", "b", "c"], [("a", "b"), ("b", "a"), ("c", "a")])
    assert any("cycle" in w.lower() for w in arch.warnings)
    assert set(simulate(arch, "a").failed) == {"a", "b", "c"}


def test_duplicate_edges_merged_with_warning():
    arch = make_arch(["a", "b"], [("a", "b", "optional"), ("a", "b", "required")])
    assert len(arch.dependencies) == 1 and arch.dependencies[0].criticality == "required"
    assert any("Duplicate dependency" in w for w in arch.warnings)


def test_unknown_failure_id():
    arch = make_arch(["a"], [])
    with pytest.raises(ArchitectureError, match="Unknown component"):
        simulate(arch, "nope")


def test_unknown_dependency_endpoint_rejected():
    with pytest.raises(ArchitectureError, match="unknown component id 'ghost'"):
        make_arch(["a"], [("a", "ghost")])


def test_self_dependency_rejected():
    with pytest.raises(ArchitectureError, match="itself"):
        make_arch(["a"], [("a", "a")])


def test_fallback_protects_only_when_healthy():
    arch = make_arch(["svc", "primary", "backup"], [("svc", "primary", "required", "backup")])
    r = simulate(arch, "primary")
    assert set(r.failed) == {"primary"} and r.fallbacks_used == [("svc", "primary", "backup")]
    both = simulate(arch, ["primary", "backup"])
    assert "svc" in both.failed


def test_deterministic_repeat():
    arch = load_sample_architecture()
    assert simulate(arch, "postgres").to_dict() == simulate(arch, "postgres").to_dict()


def test_sample_database_outage_exposes_hidden_dependent():
    arch = load_sample_architecture()
    r = simulate(arch, "postgres")
    assert r.affected_count == 7
    assert set(r.surviving) == {"redis", "message_queue"}
    # worker touches only the queue visibly, yet is hit through the shared DB
    assert "notification_worker" in r.failed and r.failed["notification_worker"].round == 1
    assert "api_gateway" in r.downstream


def test_shared_dependencies_and_blast_radius():
    arch = load_sample_architecture()
    top = shared_critical_dependencies(arch)[0]
    assert top["id"] == "postgres" and top["blast_radius"] == 6
    assert blast_radius(arch)["postgres"] == 6
