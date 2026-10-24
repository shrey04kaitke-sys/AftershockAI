import json

import pytest
import requests

from src.gemma_client import GemmaClient, GemmaError
from src.models import Architecture, ArchitectureError
from src.reporting import build_report, explanation_messages, report_to_json, report_to_markdown
from src.graph_engine import simulate
from src.interventions import Intervention, compare
from src.samples import load_sample_architecture, load_sample_description
from src.system_parser import extract_architecture, parse_model_response

DESC = "The Order Service stores orders in PostgreSQL. The Order Service calls the Payment Service."

GOOD = {
    "components": [
        {"id": "order_service", "name": "Order Service", "type": "service"},
        {"id": "postgres", "name": "PostgreSQL", "type": "database"},
        {"id": "payment_service", "name": "Payment Service", "type": "service"},
    ],
    "dependencies": [
        {"source": "order_service", "target": "postgres", "criticality": "required",
         "stated_explicitly": True, "evidence": "stores orders in PostgreSQL", "confidence": 0.9},
        {"source": "order_service", "target": "payment_service", "criticality": "required",
         "stated_explicitly": True, "evidence": "made-up quote not in text", "confidence": 7},
    ],
}


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, post=None, get=None):
        self._post, self._get, self.posts = post, get, []

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        if isinstance(self._post, Exception):
            raise self._post
        return self._post.pop(0) if isinstance(self._post, list) else self._post

    def get(self, url, timeout=None):
        if isinstance(self._get, Exception):
            raise self._get
        return self._get


def chat_reply(content, done_reason="stop"):
    return FakeResponse(200, {"message": {"content": content}, "done_reason": done_reason})


# ---- parsing / validation ----
def test_parse_assigns_provenance_from_verifiable_evidence():
    arch = parse_model_response(json.dumps(GOOD), DESC)
    by = {d.key: d for d in arch.dependencies}
    assert by[("order_service", "postgres")].provenance == "user_provided"
    assert by[("order_service", "payment_service")].provenance == "model_inferred"  # unverifiable quote
    assert by[("order_service", "payment_service")].confidence == 1.0  # clamped
    assert any("not a quote" in w for w in arch.warnings)


def test_parse_handles_code_fences_and_prose():
    arch = parse_model_response("Here you go:\n```json\n" + json.dumps(GOOD) + "\n```", DESC)
    assert len(arch.components) == 3


@pytest.mark.parametrize("raw", ["", "not json at all", '{"components": [', '[1,2]', '{"components": []}'])
def test_invalid_json_rejected(raw):
    with pytest.raises(ArchitectureError):
        parse_model_response(raw, DESC)


def test_unknown_ids_reported_clearly():
    bad = {"components": GOOD["components"],
           "dependencies": [{"source": "order_service", "target": "kafka", "criticality": "required"}]}
    with pytest.raises(ArchitectureError) as e:
        parse_model_response(json.dumps(bad), DESC)
    assert "unknown component id 'kafka'" in str(e.value)


def test_duplicate_component_ids_rejected():
    bad = {"components": [{"id": "a", "name": "A", "type": "service"}, {"id": "a", "name": "A2", "type": "service"}],
           "dependencies": []}
    with pytest.raises(ArchitectureError, match="Duplicate component id"):
        parse_model_response(json.dumps(bad), DESC)


def test_architecture_roundtrip_validates():
    arch = load_sample_architecture()
    assert Architecture.from_dict(arch.to_dict()).to_dict() == arch.to_dict()
    with pytest.raises(ArchitectureError):
        Architecture.from_dict({"components": [{"id": "a", "name": "A"}], "dependencies": [{"source": "a", "target": "zz"}]})


# ---- extraction with mocked Ollama ----
def test_extract_retries_once_on_malformed_then_succeeds():
    sess = FakeSession(post=[chat_reply("garbage"), chat_reply(json.dumps(GOOD))])
    arch = extract_architecture(GemmaClient(session=sess), DESC)
    assert len(arch.components) == 3 and len(sess.posts) == 2
    assert sess.posts[0]["model"] == "gemma4:e2b" and "format" in sess.posts[0]


def test_extract_gives_up_after_bounded_attempts():
    sess = FakeSession(post=[chat_reply("garbage"), chat_reply("still garbage"), chat_reply("x")])
    with pytest.raises(ArchitectureError, match="after 2 attempts"):
        extract_architecture(GemmaClient(session=sess), DESC)
    assert len(sess.posts) == 2


def test_model_unavailable_does_not_corrupt_state():
    state = {"architecture": load_sample_architecture()}
    before = state["architecture"].to_dict()
    sess = FakeSession(post=requests.ConnectionError("refused"))
    with pytest.raises(GemmaError, match="Cannot reach Ollama"):
        state["architecture"] = extract_architecture(GemmaClient(session=sess), DESC)
    assert state["architecture"].to_dict() == before


def test_client_error_paths():
    assert "Cannot reach" in _err(requests.ConnectionError())
    assert "did not answer" in _err(requests.Timeout())
    assert "not found" in _err(None, FakeResponse(404))
    assert "HTTP 500" in _err(None, FakeResponse(500, text="boom"))
    assert "empty response" in _err(None, chat_reply("  "))
    assert "cut off" in _err(None, chat_reply("{", done_reason="length"))


def _err(exc, resp=None):
    client = GemmaClient(session=FakeSession(post=exc or resp))
    with pytest.raises(GemmaError) as e:
        client.chat_text([{"role": "user", "content": "x"}])
    return str(e.value)


def test_status_checks():
    ok = GemmaClient(session=FakeSession(get=FakeResponse(200, {"models": [{"name": "gemma4:e2b"}]})))
    assert ok.status()[0] is True
    missing = GemmaClient(session=FakeSession(get=FakeResponse(200, {"models": [{"name": "other"}]})))
    ok_, msg = missing.status()
    assert not ok_ and "ollama pull gemma4:e2b" in msg
    down = GemmaClient(session=FakeSession(get=requests.ConnectionError()))
    assert down.status()[0] is False


# ---- reporting ----
def test_report_serialisation_and_markdown():
    arch = load_sample_architecture()
    res = simulate(arch, "payment_service")
    cmp_ = compare(arch, "payment_service", [Intervention("circuit_breaker", "order_service", "payment_service")])
    report = build_report(arch, "payment_service", res, cmp_, explanation="text", explanation_model="gemma4:e2b")
    parsed = json.loads(report_to_json(report))
    assert parsed["intervention_comparison"]["absolute_reduction"] == 2
    assert parsed["baseline_result"]["affected_count"] == 3
    assert parsed["architecture"]["dependencies"][0]["provenance"] == "sample_data"
    md = report_to_markdown(parsed)
    for needle in ("Baseline cascade", "Intervention comparison", "sample_data", "Absolute reduction: **2**",
                   "generated by gemma4:e2b", "Simplifying assumptions"):
        assert needle in md


def test_report_without_comparison_or_explanation():
    arch = load_sample_architecture()
    report = build_report(arch, "postgres", simulate(arch, "postgres"))
    assert report["intervention_comparison"] is None and report["model_explanation"] is None
    assert "Shared critical dependencies" in report_to_markdown(json.loads(report_to_json(report)))


def test_explanation_prompt_contains_only_computed_facts():
    arch = load_sample_architecture()
    msgs = explanation_messages(arch, simulate(arch, "postgres"))
    assert "Total affected: 7 of 9" in msgs[1]["content"]
    assert "Use ONLY the facts" in msgs[0]["content"]


def test_sample_description_nonempty():
    assert "PostgreSQL" in load_sample_description()


def test_model_degraded_mode_claim_is_not_trusted():
    data = json.loads(json.dumps(GOOD))
    data["components"][0]["supports_degraded_mode"] = True
    arch = parse_model_response(json.dumps(data), DESC)
    assert not any(c.supports_degraded_mode for c in arch.components)
    assert any("Model suggested degraded-mode" in w for w in arch.warnings)


def test_sample_evidence_is_quoted_from_source_description():
    from src.system_parser import _norm
    desc = _norm(load_sample_description())
    for d in load_sample_architecture().dependencies:
        assert _norm(d.evidence) in desc, f"{d.source}->{d.target}: evidence not in description"
