"""Aftershock AI - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import streamlit as st

from src.gemma_client import DEFAULT_BASE_URL, DEFAULT_MODEL, GemmaClient, GemmaError
from src.graph_engine import shared_critical_dependencies, simulate
from src.interventions import Intervention, compare
from src.models import ArchitectureError
from src.reporting import build_report, explanation_messages, report_to_json, report_to_markdown
from src.samples import load_sample_architecture, load_sample_description
from src.system_parser import extract_architecture
from src.visualization import draw_graph

st.set_page_config(page_title="Aftershock AI", page_icon="🌋", layout="wide")

SOURCE_LABELS = {
    "sample": ("🟦 SAMPLE DATA", "Bundled example - not live AI output."),
    "model": ("🟪 MODEL-GENERATED", f"Extracted live by {DEFAULT_MODEL}; model-inferred edges are unverified assumptions."),
}

ss = st.session_state
for key, default in {"arch": None, "source": None, "description": load_sample_description(),
                     "interventions": [], "failed": None, "explanation": None, "explanation_ctx": None,
                     "extract_error": None}.items():
    ss.setdefault(key, default)


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def scenario_ctx():
    return json.dumps([ss.failed, [vars(i) for i in ss.interventions]], sort_keys=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Gemma 4 status")
    base_url = st.text_input("Ollama base URL", DEFAULT_BASE_URL)
    client = GemmaClient(base_url=base_url, model=DEFAULT_MODEL)
    ok, msg = client.status()  # re-checked on every rerun
    (st.success if ok else st.error)(("🟢 " if ok else "🔴 ") + msg)
    st.caption("Model: `" + DEFAULT_MODEL + "`. Descriptions are sent only to this Ollama URL.")
    st.divider()
    st.markdown("**Legend**  \n🟥 initial failure · 🟧 cascade · 🟨 degraded · 🟩 survives  \n"
                "Solid edge = required, dashed = optional  \nEdge colour: blue user-provided, purple model-inferred, grey sample")

st.title("🌋 Aftershock AI")
st.markdown("**If this component fails, what else breaks, why - and what contains the damage?**  \n"
            "Gemma 4 extracts your architecture; a deterministic engine simulates the cascade. "
            "*Dependency-based model, not a production prediction.*")

if ss.arch is None:
    st.info("No architecture loaded yet. Load the sample in the first tab, or describe a system and extract it with Gemma 4.")

t_desc, t_graph, t_sim, t_int, t_exp = st.tabs(
    ["1 · Describe", "2 · Graph & provenance", "3 · Simulate failure", "4 · Interventions", "5 · Explain & export"])

# ---------------------------------------------------------------- 1 describe
with t_desc:
    ss.description = st.text_area("Describe your system in natural language", ss.description, height=240)
    c1, c2 = st.columns(2)
    if c1.button("⚡ Load sample architecture (instant)", use_container_width=True):
        ss.arch, ss.source, ss.failed = load_sample_architecture(), "sample", None
        ss.interventions, ss.explanation, ss.extract_error = [], None, None
        ss.description = load_sample_description()
        st.rerun()
    if c2.button("🤖 Extract dependencies with Gemma 4", type="primary", use_container_width=True, disabled=not ok):
        try:
            with st.spinner(f"Asking {DEFAULT_MODEL} locally - small models can take a minute..."):
                new_arch = extract_architecture(client, ss.description)
            ss.arch, ss.source, ss.failed = new_arch, "model", None  # only replaced on success
            ss.interventions, ss.explanation, ss.extract_error = [], None, None
            st.rerun()
        except (GemmaError, ArchitectureError) as exc:
            ss.extract_error = [str(p) for p in getattr(exc, "problems", [str(exc)])]
    if not ok:
        st.warning("Gemma extraction is disabled until Ollama and the model are reachable. You can still use the sample.")
    if ss.extract_error:
        st.error("Extraction failed - your previous architecture (if any) is unchanged:")
        for p in ss.extract_error:
            st.write("• " + p)
        st.caption("Fix the description or retry; nothing here is sample or fake output.")
    if ss.arch:
        label, note = SOURCE_LABELS[ss.source]
        st.success(f"{label}: {len(ss.arch.components)} components, {len(ss.arch.dependencies)} dependencies. {note}")

arch = ss.arch

# ---------------------------------------------------------------- 2 graph
with t_graph:
    if not arch:
        st.info("Load or extract an architecture first.")
    else:
        label, note = SOURCE_LABELS[ss.source]
        st.caption(f"{label} - {note}")
        show_fig(draw_graph(arch))
        st.subheader("Dependencies & provenance")
        st.caption("`A → B` means **A depends on B**. Provenance: user_provided = quoted from your text; "
                   "model_inferred = assumption to verify; sample_data = bundled example.")
        st.dataframe([{
            "Dependent": arch.name_of(d.source), "Depends on": arch.name_of(d.target), "Criticality": d.criticality,
            "Fallback": arch.name_of(d.fallback) if d.fallback else "", "Provenance": d.provenance,
            "Confidence (model estimate)": d.confidence, "Evidence": d.evidence} for d in arch.dependencies],
            use_container_width=True, hide_index=True)
        if arch.warnings:
            with st.expander(f"⚠ {len(arch.warnings)} warnings / assumptions", expanded=True):
                for w in arch.warnings:
                    st.write("• " + w)
        shared = shared_critical_dependencies(arch)
        if shared:
            st.subheader("Shared critical dependencies")
            st.caption("Components several services require: failing one takes all of them down, even those that look unrelated.")
            st.dataframe([{"Component": s["name"], "Directly required by": ", ".join(arch.name_of(i) for i in s["direct_dependents"]),
                           "Blast radius (others affected)": s["blast_radius"]} for s in shared],
                         use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 3 simulate
names = {c.name: c.id for c in arch.components} if arch else {}
with t_sim:
    if not arch:
        st.info("Load or extract an architecture first.")
    else:
        choice = st.selectbox("Component to fail", list(names), index=list(names.values()).index("postgres") if "postgres" in names.values() else 0)
        if st.button("💥 Simulate failure", type="primary"):
            ss.failed = names[choice]
            ss.explanation = None
        if ss.failed:
            res = simulate(arch, ss.failed)
            n = arch.name_of
            m = st.columns(4)
            m[0].metric("Affected", f"{res.affected_count} / {len(arch.components)}")
            m[1].metric("Directly affected", len(res.direct))
            m[2].metric("Downstream cascade", len(res.downstream))
            m[3].metric("Surviving", len(res.surviving))
            show_fig(draw_graph(arch, res, f"Failure of {n(ss.failed)}"))
            st.markdown(f"**Initial failure:** {n(ss.failed)}  \n"
                        f"**Directly affected:** {', '.join(map(n, res.direct)) or 'none'}  \n"
                        f"**Downstream:** {', '.join(map(n, res.downstream)) or 'none'}  \n"
                        f"**Degraded but operational:** {', '.join(map(n, res.degraded)) or 'none'}  \n"
                        f"**Surviving:** {', '.join(map(n, res.surviving)) or 'none'}")
            with st.expander("Why did each component fail?", expanded=True):
                for cid, info in res.failed.items():
                    if info.round:
                        st.write(f"• **{n(cid)}** - round {info.round}: requires failed {', '.join(map(n, info.causes))}")
            hidden = [c for c in res.direct if not any(d.source == c and d.target == ss.failed for d in arch.dependencies)]
            if res.downstream or hidden:
                st.info("💡 Components that never talk to the failed one still fall through shared dependencies: "
                        + ", ".join(map(n, res.downstream)))
        else:
            st.info("Pick a component and press **Simulate failure**.")

# ---------------------------------------------------------------- 4 interventions
with t_int:
    if not arch:
        st.info("Load or extract an architecture first.")
    elif not ss.failed:
        st.info("Run a simulation in tab 3 first - interventions are compared against that failure.")
    else:
        n = arch.name_of
        st.markdown(f"Comparing against failure of **{n(ss.failed)}**. Same propagation rules for baseline and intervention.")
        with st.expander("Declare degraded-mode support (your assertion; required for circuit breakers)"):
            st.caption("Tick only components that can genuinely keep serving, in reduced form, when a dependency is unavailable.")
            for comp in arch.components:
                comp.supports_degraded_mode = st.checkbox(
                    comp.name, comp.supports_degraded_mode, key=f"deg_{comp.id}",
                    help=comp.degraded_behavior or None)
        edge_labels = {f"{n(d.source)} → {n(d.target)}": d for d in arch.dependencies}
        kind = st.radio("Intervention", ["circuit_breaker", "isolate", "fallback"], horizontal=True,
                        format_func={"circuit_breaker": "Circuit breaker (degrade instead of fail)",
                                     "isolate": "Isolate dependency (remove edge)",
                                     "fallback": "Add fallback component"}.get)
        edge = st.selectbox("Dependency (dependent → depends on)", list(edge_labels))
        fb = None
        if kind == "fallback":
            d = edge_labels[edge]
            opts = {c.name: c.id for c in arch.components if c.id not in (d.source, d.target)}
            fb = opts[st.selectbox("Fallback component", list(opts))]
        if kind == "circuit_breaker" and not arch.get(edge_labels[edge].source).supports_degraded_mode:
            st.caption("⚠ This dependent does not declare a degraded mode; the breaker will be rejected as not applicable.")
        a, b = st.columns(2)
        if a.button("➕ Add intervention"):
            d = edge_labels[edge]
            ss.interventions.append(Intervention(kind, d.source, d.target, fb))
            ss.explanation = None
        if b.button("🗑 Clear interventions"):
            ss.interventions, ss.explanation = [], None
        for iv in ss.interventions:
            st.write("• " + iv.describe(arch))
        if ss.interventions:
            try:
                cmp_ = compare(arch, ss.failed, ss.interventions)
            except ArchitectureError as exc:
                st.error("; ".join(exc.problems))
                if st.button("Remove last intervention"):
                    ss.interventions.pop()
                    st.rerun()
            else:
                (st.success if cmp_.helps else st.warning)(cmp_.verdict())
                m = st.columns(4)
                m[0].metric("Baseline affected", cmp_.baseline.affected_count)
                m[1].metric("With interventions", cmp_.scenario.affected_count)
                m[2].metric("Absolute reduction", cmp_.reduction)
                m[3].metric("Protected", len(cmp_.protected))
                g1, g2 = st.columns(2)
                with g1:
                    show_fig(draw_graph(arch, cmp_.baseline, "Baseline"))
                with g2:
                    from src.interventions import apply_interventions
                    show_fig(draw_graph(apply_interventions(arch, ss.interventions), cmp_.scenario, "With interventions"))
                st.markdown(f"**Protected:** {', '.join(map(n, cmp_.protected)) or 'none'}  \n"
                            f"**Still affected:** {', '.join(map(n, cmp_.still_affected)) or 'none'}")
                st.caption("Assumes the intervention works perfectly as declared (instant, no config or load effects).")
        else:
            st.info("Add at least one intervention to compare. Tip: with the sample, fail **Payment Service** and add a circuit breaker on Order Service → Payment Service.")

# ---------------------------------------------------------------- 5 explain & export
with t_exp:
    if not arch or not ss.failed:
        st.info("Run a simulation first.")
    else:
        res = simulate(arch, ss.failed)
        cmp_ = None
        if ss.interventions:
            try:
                cmp_ = compare(arch, ss.failed, ss.interventions)
            except ArchitectureError:
                pass
        if st.button("🤖 Explain with Gemma 4", disabled=not ok):
            try:
                with st.spinner("Gemma 4 is explaining the computed results..."):
                    ss.explanation = client.chat_text(explanation_messages(arch, res, cmp_))
                    ss.explanation_ctx = scenario_ctx()
            except GemmaError as exc:
                st.error(f"Explanation failed: {exc}")
        if ss.explanation and ss.explanation_ctx == scenario_ctx():
            st.markdown("🟪 **Model-generated explanation** (text only; numbers come from the deterministic engine)")
            st.write(ss.explanation)
        else:
            ss.explanation = None
        report = build_report(arch, ss.failed, res, cmp_, ss.explanation, DEFAULT_MODEL if ss.explanation else None)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ JSON report", report_to_json(report), "aftershock_report.json", "application/json", use_container_width=True)
        d2.download_button("⬇ Markdown report", report_to_markdown(report), "aftershock_report.md", "text/markdown", use_container_width=True)
        with st.expander("Preview Markdown report"):
            st.markdown(report_to_markdown(report))
