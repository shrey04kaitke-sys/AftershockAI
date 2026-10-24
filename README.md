# 🌋 Aftershock AI

**If this component fails, what else breaks, why does it break, and what intervention could contain the damage?**

Aftershock AI is a local cascading-failure simulator for software architectures. You describe a system in plain English; **Gemma 4** (via Ollama) turns it into a validated dependency graph; a **deterministic engine** simulates how one failure propagates; you compare interventions against the baseline.

> **The insight it is built to show:** a component that looks healthy and isolated may still be vulnerable because it shares a critical dependency with other services. In the sample, the *Notification Worker* is wired visibly only to the *Message Queue* — yet a PostgreSQL outage takes it down too.

> ⚠ This is a **dependency-based failure-propagation model**, not a production-accurate prediction. It has no load, capacity, timeout, retry or probability terms.

## Why Gemma 4

| Step | Who does it | Why |
|---|---|---|
| Architecture text → components/dependencies | **Gemma 4** (`gemma4:e2b`) | Free-form prose can't be parsed with rules. |
| Validation, provenance, simulation, intervention maths | **Deterministic code** | Results must be reproducible and auditable; an LLM never decides a calculated result. |
| Plain-language explanation of results | **Gemma 4** | Given only the computed facts, with instructions not to invent anything. |

If Ollama or the model is unavailable the app says so and disables model actions; it never fakes model output. The bundled sample is always labelled **SAMPLE DATA**.

## Architecture & data flow

```
description ──► gemma_client ──► system_parser ──► Architecture (validated, with provenance)
                (Ollama /api/chat, JSON schema)        │
                                                       ▼
                 graph_engine.simulate  ◄── interventions.apply (on a copy)
                         │                         │
                         └──── compare(baseline, scenario) ───► reporting (JSON + Markdown)
                                      │                              ▲
                                      └──► Gemma explanation (text only) ┘      app.py (Streamlit) renders it all
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI (5 tabs) |
| `src/models.py` | Schema dataclasses and `ArchitectureError` |
| `src/gemma_client.py` | Ollama client: status check, timeouts, clear errors |
| `src/system_parser.py` | Prompt, JSON parsing, validation, provenance assignment, bounded retry |
| `src/graph_engine.py` | Deterministic cascade simulation, blast radius, shared dependencies |
| `src/interventions.py` | Isolate / circuit breaker / fallback + baseline comparison |
| `src/reporting.py` | Explanation prompt, JSON and Markdown reports |
| `src/visualization.py` | Layered matplotlib graph |
| `examples/` | Sample architecture (`ecommerce.json`) and its source text |
| `tests/` | pytest suite (Ollama mocked) |

## Schema and provenance

- **Edge direction:** `source → target` means **"source depends on target"**. Failures flow against the arrow.
- Component: `id`, `name`, `type`, `supports_degraded_mode`, `degraded_behavior`, `provenance`.
- Dependency: `source`, `target`, `criticality` (`required` | `optional`), optional `fallback` component, `provenance`, `evidence`, `confidence`.
- **Provenance:** `user_provided`, `model_inferred`, `sample_data`. Gemma's claim that an edge is explicit is only honoured if its `evidence` is a **verifiable quote from your text**; otherwise the edge is `model_inferred` and a warning is shown. `confidence` is the model's own estimate, not a calibrated probability.
- **Degraded mode is never taken from the model.** Gemma's suggestions appear as warnings; you declare which components can run degraded in the Interventions tab (needed for circuit breakers). Live testing showed the model over-claims this (7 of 9 components).
- Validation errors (unknown IDs, duplicate components, self-dependencies, bad enums, malformed/truncated JSON) are reported as readable lists. Duplicate edges are merged with a warning; cycles are allowed with a warning.

## Simulation model

1. An edge is **broken** if its target has failed *and* its declared fallback (if any) has not also failed.
2. A component **fails** if any `required` edge is broken.
3. A component is **degraded** (still operational) if only `optional` edges are broken. Degradation does not propagate.
4. Rounds repeat to a fixed point (so cycles terminate). Round 0 = initial failure, 1 = directly affected, 2+ = downstream cascade.
5. Order is fixed, so results are reproducible.

**Interventions** run on a copy, using the same rules:
- *Isolate*: remove the edge (e.g. make the call asynchronous).
- *Circuit breaker*: required → optional on that edge; **only applicable if the dependent declares degraded mode** (otherwise rejected).
- *Fallback*: declare a substitute; protects only if the substitute itself hasn't failed (a fallback sharing the failed dependency gives no benefit — tested).

Interventions are assumed to work perfectly and instantly. "No reduction" is reported honestly.

## Install (Windows PowerShell)

```powershell
cd "C:\Users\shrey\OneDrive\Desktop\BOOKS NAME & IMAGES\00_Home\03_projects\AftershockAI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify Ollama and the model:

```powershell
ollama serve            # skip if the Ollama app is already running
ollama list             # must show gemma4:e2b
ollama pull gemma4:e2b  # only if missing
```

Set `OLLAMA_BASE_URL` to use a non-default address (default `http://localhost:11434`).

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
python -m pytest -q
```

Unit tests mock Ollama and need no network or running model. `tests/test_app_smoke.py` drives the Streamlit UI headlessly (it tries `localhost:11434` for the status badge and copes if it's down); it takes ~30-45 s. The suite currently has 46 tests: engine, interventions, parser/client (Ollama mocked), reports, and headless UI flows including the Ollama-unavailable case.

## Demo walkthrough (~3 minutes)

1. Open the app; point at the sidebar status badge (green = `gemma4:e2b` reachable).
2. **Tab 1:** click **Load sample architecture** (instant, labelled SAMPLE DATA). Optionally click **Extract dependencies with Gemma 4** on the same text (~20 s on the dev machine) and compare provenance in tab 2.
3. **Tab 2:** show the graph, the provenance table, and *Shared critical dependencies* — PostgreSQL is directly required by 5 components (blast radius 6).
4. **Tab 3:** fail **PostgreSQL** → 7 of 9 affected; only Redis and the Message Queue survive. Note the Notification Worker falls despite only visibly touching the queue.
5. Fail **Payment Service** → 3 affected (Payment, Order, API Gateway).
6. **Tab 4:** add *Circuit breaker* on **Order Service → Payment Service** → 3 → 1 affected, Order Service degraded instead of failed, reduction 2. (Try a breaker on the API Gateway to see it rejected as not applicable.)
7. **Tab 5:** click **Explain with Gemma 4**, then download the JSON/Markdown reports.

## Development note

This project was developed with the help of an AI coding assistant; the repository's sole credited author is the account owner.

## Privacy and network behaviour

Descriptions and simulation facts are sent only to the Ollama URL you configure (default: local machine). The app makes no other network calls and uses no API keys. Streamlit itself may check for updates/usage stats per its own settings; see Streamlit's documentation to disable.

## Known limitations

- Reachability model: no capacity, load, latency, retries, partial failure or probabilities.
- Extraction quality depends on a small local model. In a live run on the sample text `gemma4:e2b` extracted all 9 components and 13 dependencies but labelled some required edges (e.g. Order Service → Payment Service) as optional; such edges are shown as `model_inferred`. Always review the provenance table, and use the bundled sample for reliable demo numbers.
- Single initial failure in the UI (the engine accepts several).
- Edge criticality and degraded-mode support are assertions, not measurements.
- Intervention benefit assumes ideal behaviour.
- Not novel, not production-ready; no evaluation or benchmark results are claimed.
