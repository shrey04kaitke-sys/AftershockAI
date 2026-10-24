from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _click(at, label):
    next(b for b in at.button if label in b.label).click().run()


def test_app_sample_flow_end_to_end():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    _click(at, "Load sample")
    assert not at.exception and any("SAMPLE DATA" in s.value for s in at.success)
    at.selectbox[0].select("Payment Service")
    _click(at, "Simulate failure")
    assert not at.exception
    assert any(m.label == "Affected" and m.value.startswith("3 /") for m in at.metric)
    next(s for s in at.selectbox if s.label.startswith("Dependency")).select("Order Service → Payment Service")
    at.run()
    _click(at, "Add intervention")
    assert not at.exception
    assert any(m.label == "Absolute reduction" and m.value == "2" for m in at.metric)


def test_app_with_ollama_unavailable_keeps_sample_working():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value("http://127.0.0.1:9").run()  # nothing listens here
    assert not at.exception
    assert any("not reachable" in e.value for e in at.sidebar.error)
    extract = next(b for b in at.button if "Extract dependencies" in b.label)
    assert extract.disabled
    _click(at, "Load sample")
    assert not at.exception and any("SAMPLE DATA" in s.value for s in at.success)
    at.selectbox[0].select("PostgreSQL")
    _click(at, "Simulate failure")
    assert any(m.label == "Affected" and m.value == "7 / 9" for m in at.metric)
    assert next(b for b in at.button if "Explain with Gemma" in b.label).disabled


def test_download_buttons_present_after_simulation():
    at = AppTest.from_file(APP, default_timeout=60).run()
    _click(at, "Load sample")
    at.selectbox[0].select("Payment Service")
    _click(at, "Simulate failure")
    labels = [e.proto.label for e in at.get("download_button")]
    assert any("JSON report" in l for l in labels) and any("Markdown report" in l for l in labels)
