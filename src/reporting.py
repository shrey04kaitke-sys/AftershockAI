"""Gemma explanation prompt plus JSON / Markdown report generation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .graph_engine import SimulationResult, shared_critical_dependencies
from .interventions import Comparison
from .models import Architecture

SIMPLIFICATIONS = [
    "Dependency-based propagation only: no load, capacity, timeout, retry or probability modelling.",
    "A component fails if any 'required' dependency fails (unless its declared fallback is healthy).",
    "'Optional' dependencies only degrade a component; degradation does not propagate.",
    "Results are as accurate as the dependency data; model-inferred edges are unverified assumptions.",
    "This is not a prediction of production behaviour.",
]


def explanation_messages(arch: Architecture, result: SimulationResult,
                         comparison: Comparison | None = None) -> list[dict[str, str]]:
    """Prompt for Gemma: explain ONLY the supplied, already-computed facts."""
    n = arch.name_of
    facts = [
        f"Initial failure: {', '.join(n(c) for c in result.initial)}",
        f"Directly affected: {', '.join(n(c) for c in result.direct) or 'none'}",
        f"Downstream cascade: {', '.join(n(c) for c in result.downstream) or 'none'}",
        f"Still operational: {', '.join(n(c) for c in result.surviving) or 'none'}",
        f"Degraded but operational: {', '.join(n(c) for c in result.degraded) or 'none'}",
        f"Total affected: {result.affected_count} of {len(arch.components)}",
    ]
    for cid, info in result.failed.items():
        if info.round:
            facts.append(f"{n(cid)} failed in round {info.round} because it requires failed: "
                         + ", ".join(n(c) for c in info.causes))
    if comparison:
        facts += [
            "Interventions: " + "; ".join(i.describe(arch) for i in comparison.interventions),
            f"With interventions affected: {comparison.scenario.affected_count} (baseline {comparison.baseline.affected_count})",
            f"Protected: {', '.join(n(c) for c in comparison.protected) or 'none'}",
            f"Verdict: {comparison.verdict()}",
        ]
    return [
        {"role": "system", "content": (
            "You are a reliability engineer. Explain the simulation results below in 4-6 sentences for a developer. "
            "Use ONLY the facts provided. Do not invent numbers, components, causes or probabilities. "
            "Mention that this is a dependency-based model, not a production prediction.")},
        {"role": "user", "content": "Facts:\n- " + "\n- ".join(facts)},
    ]


def build_report(arch: Architecture, failed_component: str, result: SimulationResult,
                 comparison: Comparison | None = None, explanation: str | None = None,
                 explanation_model: str | None = None) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "Aftershock AI",
        "edge_semantics": "source -> target means 'source depends on target'; failure propagates from target to source.",
        "architecture": arch.to_dict(),
        "scenario": {"failed_component": failed_component},
        "baseline_result": result.to_dict(),
        "shared_critical_dependencies": shared_critical_dependencies(arch),
        "intervention_comparison": comparison.to_dict(arch) if comparison else None,
        "model_explanation": (
            {"text": explanation, "model": explanation_model,
             "note": "Generated text; not part of the deterministic calculation."} if explanation else None),
        "simplifying_assumptions": SIMPLIFICATIONS,
    }


def _names(arch: Architecture, ids: list[str]) -> str:
    return ", ".join(arch.name_of(i) for i in ids) if ids else "none"


def report_to_markdown(report: dict[str, Any]) -> str:
    arch = Architecture.from_dict(report["architecture"], _validate=False)
    res = report["baseline_result"]
    L: list[str] = ["# Aftershock AI report", "",
                    f"_Generated {report['generated_at']}_", "",
                    f"**System:** {arch.name}  ", f"**Failed component:** {arch.name_of(report['scenario']['failed_component'])}", "",
                    "> Edge `A -> B` means *A depends on B*; failures propagate from B to A.", "",
                    "## Components", ""]
    L += [f"- **{c.name}** (`{c.id}`, {c.type}){' - supports degraded mode' if c.supports_degraded_mode else ''} [{c.provenance}]"
          for c in arch.components]
    L += ["", "## Dependencies and provenance", "",
          "| Dependent | Depends on | Criticality | Fallback | Provenance | Confidence | Evidence |",
          "|---|---|---|---|---|---|---|"]
    for d in arch.dependencies:
        conf = f"{d.confidence:.2f} (model estimate)" if d.confidence is not None else "-"
        L.append(f"| {arch.name_of(d.source)} | {arch.name_of(d.target)} | {d.criticality} | "
                 f"{arch.name_of(d.fallback) if d.fallback else '-'} | {d.provenance} | {conf} | {d.evidence.replace('|', '/') or '-'} |")
    if arch.warnings:
        L += ["", "### Warnings / assumptions", ""] + [f"- {w}" for w in arch.warnings]
    waves = res["waves"]
    L += ["", "## Baseline cascade", "",
          f"- Initial failure: {_names(arch, waves[0])}",
          f"- Directly affected: {_names(arch, waves[1] if len(waves) > 1 else [])}",
          f"- Downstream cascade: {_names(arch, [c for w in waves[2:] for c in w])}",
          f"- Degraded but operational: {_names(arch, list(res['degraded']))}",
          f"- Surviving: {_names(arch, res['surviving'])}",
          f"- **Total affected: {res['affected_count']} of {len(arch.components)}**", ""]
    for cid, info in res["failed"].items():
        if info["round"]:
            L.append(f"  - {arch.name_of(cid)} (round {info['round']}) requires failed: {_names(arch, info['causes'])}")
    shared = report.get("shared_critical_dependencies") or []
    if shared:
        L += ["", "## Shared critical dependencies", ""]
        L += [f"- **{s['name']}**: required by {_names(arch, s['direct_dependents'])}; blast radius {s['blast_radius']}" for s in shared]
    cmp_ = report.get("intervention_comparison")
    if cmp_:
        b, w = cmp_["baseline"], cmp_["with_interventions"]
        L += ["", "## Intervention comparison", ""]
        L += [f"- {i['description']}" for i in cmp_["interventions"]]
        L += ["", "| | Affected | Components |", "|---|---|---|",
              f"| Baseline | {b['affected_count']} | {_names(arch, [c for x in b['waves'] for c in x])} |",
              f"| With interventions | {w['affected_count']} | {_names(arch, [c for x in w['waves'] for c in x])} |", "",
              f"- Protected: {_names(arch, cmp_['protected'])}",
              f"- Still affected: {_names(arch, cmp_['still_affected'])}",
              f"- Absolute reduction: **{cmp_['absolute_reduction']}**",
              f"- Verdict: {cmp_['verdict']}"]
    expl = report.get("model_explanation")
    if expl:
        L += ["", f"## Model explanation (generated by {expl['model']}; not a calculation)", "", expl["text"]]
    L += ["", "## Simplifying assumptions", ""] + [f"- {s}" for s in report["simplifying_assumptions"]]
    return "\n".join(L) + "\n"


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
