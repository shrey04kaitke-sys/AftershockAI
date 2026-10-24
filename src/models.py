"""Architecture schema.

Edge direction (enforced everywhere): a Dependency with ``source=A, target=B``
means **"A depends on B"** (A calls / requires B). Failures therefore propagate
against the arrow: when B fails, A may fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

PROVENANCE = ("user_provided", "model_inferred", "sample_data")
CRITICALITY = ("required", "optional")
COMPONENT_TYPES = (
    "gateway", "service", "database", "cache", "queue", "worker",
    "external", "infrastructure", "other",
)


class ArchitectureError(ValueError):
    """Raised when architecture data is invalid. ``problems`` lists each issue."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass
class Component:
    id: str
    name: str
    type: str = "service"
    # Declared behaviour: can this component keep serving (degraded) when an
    # optional/breaker-protected dependency is unavailable?
    supports_degraded_mode: bool = False
    degraded_behavior: str = ""
    provenance: str = "user_provided"


@dataclass
class Dependency:
    source: str  # the dependent
    target: str  # what it depends on
    criticality: str = "required"  # required: target failure fails source; optional: only degrades it
    fallback: str | None = None  # component id that substitutes for target if target fails
    provenance: str = "user_provided"
    evidence: str = ""
    confidence: float | None = None  # model estimate, NOT a calibrated probability

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass
class Architecture:
    components: list[Component] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    name: str = "Untitled system"

    def component_ids(self) -> list[str]:
        return [c.id for c in self.components]

    def get(self, cid: str) -> Component:
        for c in self.components:
            if c.id == cid:
                return c
        raise ArchitectureError(f"Unknown component id '{cid}'")

    def name_of(self, cid: str) -> str:
        try:
            return self.get(cid).name
        except ArchitectureError:
            return cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": [asdict(c) for c in self.components],
            "dependencies": [asdict(d) for d in self.dependencies],
            "warnings": list(self.warnings),
        }

    def copy(self) -> "Architecture":
        return Architecture.from_dict(self.to_dict(), _validate=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any], _validate: bool = True) -> "Architecture":
        arch = cls(
            name=str(data.get("name") or "Untitled system"),
            components=[Component(**c) for c in data.get("components", [])],
            dependencies=[Dependency(**d) for d in data.get("dependencies", [])],
            warnings=list(data.get("warnings", [])),
        )
        if _validate:
            from .system_parser import validate_architecture
            return validate_architecture(arch)
        return arch
