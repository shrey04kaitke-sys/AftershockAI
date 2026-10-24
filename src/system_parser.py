"""Prompting, JSON parsing, validation and normalisation of architecture data."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import networkx as nx

from .models import (
    COMPONENT_TYPES, CRITICALITY, PROVENANCE, Architecture, ArchitectureError,
    Component, Dependency,
)

ARCHITECTURE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(COMPONENT_TYPES)},
                    "supports_degraded_mode": {"type": "boolean"},
                    "degraded_behavior": {"type": "string"},
                },
                "required": ["id", "name", "type"],
            },
        },
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "criticality": {"type": "string", "enum": list(CRITICALITY)},
                    "fallback": {"type": ["string", "null"]},
                    "stated_explicitly": {"type": "boolean"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["source", "target", "criticality"],
            },
        },
    },
    "required": ["components", "dependencies"],
}

SYSTEM_PROMPT = (
    "You extract software architecture into JSON. Output ONLY JSON.\n\n"
    "Schema:\n"
    "- components[]: id (short snake_case, unique), name, type (one of: " + ", ".join(COMPONENT_TYPES) + "),\n"
    "  supports_degraded_mode (true ONLY if the text says the component can keep working without a dependency),\n"
    "  degraded_behavior (short text or \"\").\n"
    "- dependencies[]: source, target, criticality, fallback, stated_explicitly, evidence, confidence.\n\n"
    "EDGE DIRECTION: {source: A, target: B} means \"A DEPENDS ON B\" (A calls or needs B).\n"
    "Example: \"orders are stored in PostgreSQL\" -> source=order_service, target=postgres.\n"
    "Dependencies point FROM the dependent TO the thing it needs, never the reverse.\n\n"
    "criticality: \"required\" if the source cannot work without the target; \"optional\" if the text says it only\n"
    "improves performance/features or can work without it (e.g. a cache).\n"
    "fallback: the id of another component that substitutes when the target fails, only if the text says so, else null.\n"
    "stated_explicitly: true only if the text states this dependency directly; false if you are guessing.\n"
    "evidence: for explicit dependencies, an EXACT short quote copied from the text; otherwise \"\".\n"
    "confidence: your own 0-1 estimate (not a calibrated probability).\n"
    "Do not invent components that the text does not mention. All ids used in dependencies must appear in components."
)


def build_messages(description: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Architecture description:\n\n{description.strip()}"},
    ]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


def extract_json_text(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences / surrounding prose."""
    if not isinstance(raw, str) or not raw.strip():
        raise ArchitectureError("Model returned an empty response.")
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ArchitectureError("Model response was not valid JSON (no JSON object found).")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ArchitectureError(f"Model response was malformed or truncated JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ArchitectureError("Model JSON must be an object with 'components' and 'dependencies'.")
    return data


def validate_architecture(arch: Architecture) -> Architecture:
    """Validate (and lightly normalise) an architecture. Raises ArchitectureError listing all problems.

    - duplicate component ids / unknown ids / self-dependencies / bad enums are errors
    - duplicate edges are merged (strictest criticality kept) with a warning
    - cycles are allowed but warned about
    """
    problems: list[str] = []
    warnings = list(arch.warnings)
    if not arch.components:
        problems.append("Architecture has no components.")

    seen: set[str] = set()
    for c in arch.components:
        if not c.id or not re.fullmatch(r"[a-z0-9_]+", c.id):
            problems.append(f"Component id '{c.id}' must be non-empty lowercase letters, digits or underscores.")
        if c.id in seen:
            problems.append(f"Duplicate component id '{c.id}'.")
        seen.add(c.id)
        if not c.name:
            problems.append(f"Component '{c.id}' has no name.")
        if c.type not in COMPONENT_TYPES:
            warnings.append(f"Component '{c.id}': unknown type '{c.type}', treated as 'other'.")
            c.type = "other"
        if c.provenance not in PROVENANCE:
            problems.append(f"Component '{c.id}': invalid provenance '{c.provenance}'.")

    merged: dict[tuple[str, str], Dependency] = {}
    for d in arch.dependencies:
        bad = False
        for end, label in ((d.source, "source"), (d.target, "target")):
            if end not in seen:
                problems.append(f"Dependency {d.source} -> {d.target}: unknown component id '{end}' ({label}).")
                bad = True
        if d.source == d.target:
            problems.append(f"Dependency {d.source} -> {d.target}: a component cannot depend on itself.")
            bad = True
        if d.criticality not in CRITICALITY:
            problems.append(f"Dependency {d.source} -> {d.target}: criticality must be one of {CRITICALITY}, got '{d.criticality}'.")
            bad = True
        if d.provenance not in PROVENANCE:
            problems.append(f"Dependency {d.source} -> {d.target}: invalid provenance '{d.provenance}'.")
            bad = True
        if d.fallback is not None:
            if d.fallback not in seen:
                problems.append(f"Dependency {d.source} -> {d.target}: unknown fallback id '{d.fallback}'.")
                bad = True
            elif d.fallback in (d.source, d.target):
                problems.append(f"Dependency {d.source} -> {d.target}: fallback cannot be the source or target itself.")
                bad = True
        if d.confidence is not None:
            try:
                d.confidence = min(1.0, max(0.0, float(d.confidence)))
            except (TypeError, ValueError):
                d.confidence = None
        if bad:
            continue
        if d.key in merged:
            prev = merged[d.key]
            warnings.append(f"Duplicate dependency {d.source} -> {d.target} merged (strictest criticality kept).")
            if d.criticality == "required":
                prev.criticality = "required"
            prev.fallback = prev.fallback or d.fallback
        else:
            merged[d.key] = d
    if problems:
        raise ArchitectureError(problems)

    arch.dependencies = list(merged.values())
    g = nx.DiGraph([(d.source, d.target) for d in arch.dependencies])
    for cycle in nx.simple_cycles(g):
        warnings.append("Dependency cycle: " + " -> ".join(cycle + [cycle[0]]))
    arch.warnings = list(dict.fromkeys(warnings))
    return arch


def parse_model_response(raw: str, description: str, name: str = "Model-extracted system") -> Architecture:
    """Turn raw model text into a validated Architecture, assigning provenance deterministically.

    An edge is only marked ``user_provided`` when the model's evidence string is a verifiable quote
    from the user's description; otherwise it is ``model_inferred``.
    """
    data = extract_json_text(raw)
    comps_raw, deps_raw = data.get("components"), data.get("dependencies")
    if not isinstance(comps_raw, list) or not isinstance(deps_raw, list):
        raise ArchitectureError("Model JSON must contain 'components' and 'dependencies' arrays.")
    norm_desc = _norm(description)
    warnings: list[str] = []
    suggestions: list[str] = []

    components: list[Component] = []
    alias: dict[str, str] = {}
    for item in comps_raw:
        if not isinstance(item, dict) or not (item.get("id") or item.get("name")):
            raise ArchitectureError("Each component must be an object with an 'id' and 'name'.")
        cname = str(item.get("name") or item.get("id")).strip()
        cid = slugify(item.get("id") or cname)
        alias[slugify(cname)] = cid
        alias[cid] = cid
        mentioned = _norm(cname) in norm_desc
        suggested = bool(item.get("supports_degraded_mode", False))
        if suggested:
            suggestions.append(cname)
        components.append(Component(
            id=cid, name=cname, type=str(item.get("type") or "other").lower(),
            supports_degraded_mode=False,  # never trusted from the model; the user must confirm it
            degraded_behavior=str(item.get("degraded_behavior") or ""),
            provenance="user_provided" if mentioned else "model_inferred",
        ))
        if not mentioned:
            warnings.append(f"Component '{cname}' is not named in your description (model-inferred).")

    def resolve(ref: Any) -> str:
        s = slugify(ref)
        return alias.get(s, s)

    deps: list[Dependency] = []
    for item in deps_raw:
        if not isinstance(item, dict) or "source" not in item or "target" not in item:
            raise ArchitectureError("Each dependency must be an object with 'source' and 'target'.")
        evidence = str(item.get("evidence") or "").strip()
        explicit = bool(item.get("stated_explicitly", False))
        verified = bool(evidence) and _norm(evidence) in norm_desc
        if explicit and not verified:
            warnings.append(f"Dependency {item['source']} -> {item['target']}: model claimed it was explicit but its evidence is not a quote from your text; treated as inferred.")
        fb = item.get("fallback")
        conf = item.get("confidence")
        deps.append(Dependency(
            source=resolve(item["source"]), target=resolve(item["target"]),
            criticality=str(item.get("criticality") or "required").lower(),
            fallback=resolve(fb) if fb else None,
            provenance="user_provided" if (explicit and verified) else "model_inferred",
            evidence=evidence if verified else "",
            confidence=conf if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None,
        ))

    if suggestions:
        warnings.append("Model suggested degraded-mode support for: " + ", ".join(suggestions)
                        + ". Not applied - declare it yourself in the Interventions tab before using circuit breakers.")
    return validate_architecture(Architecture(components=components, dependencies=deps, warnings=warnings, name=name))


def extract_architecture(client: Any, description: str, max_attempts: int = 2,
                         on_attempt: Callable[[int], None] | None = None) -> Architecture:
    """Call Gemma via ``client.chat_json`` and validate. Retries only on parse/validation failure (bounded).

    Connection/timeouts (GemmaError) propagate immediately; they are not retried.
    """
    if not description or not description.strip():
        raise ArchitectureError("Please enter a system description first.")
    messages = build_messages(description)
    last: ArchitectureError | None = None
    for attempt in range(1, max_attempts + 1):
        if on_attempt:
            on_attempt(attempt)
        raw = client.chat_json(messages, ARCHITECTURE_JSON_SCHEMA)
        try:
            return parse_model_response(raw, description)
        except ArchitectureError as exc:
            last = exc
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That output was rejected: " + "; ".join(exc.problems)
                 + ". Return corrected JSON only, keeping dependency direction 'source depends on target'."},
            ]
    raise ArchitectureError([f"Model output failed validation after {max_attempts} attempts."] + (last.problems if last else []))
