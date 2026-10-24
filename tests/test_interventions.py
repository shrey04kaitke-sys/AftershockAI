import pytest

from src.interventions import Intervention, apply_interventions, compare
from src.models import ArchitectureError
from src.samples import DEMO_SCENARIOS, load_sample_architecture
from tests.helpers import make_arch


def test_sample_breaker_reduces_payment_cascade():
    arch = load_sample_architecture()
    sc = DEMO_SCENARIOS["payment_outage_with_breaker"]
    cmp_ = compare(arch, sc["fail"], [Intervention(**i) for i in sc["interventions"]])
    assert cmp_.baseline.affected_count == 3
    assert cmp_.scenario.affected_count == 1
    assert cmp_.protected == ["api_gateway", "order_service"]
    assert cmp_.still_affected == ["payment_service"]
    assert cmp_.reduction == 2 and cmp_.helps
    assert cmp_.scenario.degraded == {"order_service": ["payment_service"]}


def test_original_architecture_not_mutated():
    arch = load_sample_architecture()
    before = arch.to_dict()
    compare(arch, "payment_service", [Intervention("circuit_breaker", "order_service", "payment_service")])
    assert arch.to_dict() == before


def test_isolate_removes_edge():
    arch = make_arch(["a", "b", "c"], [("a", "b"), ("b", "c")])
    cmp_ = compare(arch, "c", [Intervention("isolate", "b", "c")])
    assert cmp_.baseline.affected_count == 3 and cmp_.scenario.affected_count == 1
    assert cmp_.protected == ["a", "b"] and cmp_.reduction == 2


def test_breaker_rejected_without_declared_degraded_mode():
    arch = make_arch(["a", "b"], [("a", "b")])
    with pytest.raises(ArchitectureError, match="not applicable"):
        apply_interventions(arch, [Intervention("circuit_breaker", "a", "b")])


def test_fallback_intervention_protects_when_fallback_healthy():
    arch = make_arch(["a", "b", "c"], [("a", "b")])
    cmp_ = compare(arch, "b", [Intervention("fallback", "a", "b", fallback="c")])
    assert cmp_.protected == ["a"] and cmp_.reduction == 1
    assert cmp_.scenario.fallbacks_used == [("a", "b", "c")]


def test_fallback_that_shares_the_failed_dependency_gives_no_benefit():
    arch = make_arch(["a", "b", "c"], [("a", "b"), ("c", "b")])  # c also needs b
    cmp_ = compare(arch, "b", [Intervention("fallback", "a", "b", fallback="c")])
    assert "c" in cmp_.scenario.failed and "a" in cmp_.scenario.failed
    assert cmp_.reduction == 0 and cmp_.protected == []


def test_missing_edge_and_bad_fallback_errors():
    arch = make_arch(["a", "b"], [])
    with pytest.raises(ArchitectureError, match="No dependency"):
        apply_interventions(arch, [Intervention("isolate", "a", "b")])
    arch = make_arch(["a", "b"], [("a", "b")])
    with pytest.raises(ArchitectureError, match="does not exist"):
        apply_interventions(arch, [Intervention("fallback", "a", "b", fallback="zzz")])


def test_no_benefit_is_reported_honestly():
    arch = make_arch(["a", "b", "c"], [("a", "b"), ("a", "c")])
    cmp_ = compare(arch, "b", [Intervention("isolate", "a", "c")])  # irrelevant edge
    assert cmp_.reduction == 0 and not cmp_.helps and "No reduction" in cmp_.verdict()
