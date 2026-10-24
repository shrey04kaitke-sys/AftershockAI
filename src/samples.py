"""Bundled sample architecture (clearly labelled sample data, never live model output)."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Architecture

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Scenarios used in the README demo walkthrough.
DEMO_SCENARIOS = {
    "database_outage": {"fail": "postgres", "interventions": []},
    "payment_outage_with_breaker": {
        "fail": "payment_service",
        "interventions": [{"kind": "circuit_breaker", "source": "order_service", "target": "payment_service"}],
    },
}


def load_sample_architecture() -> Architecture:
    data = json.loads((EXAMPLES_DIR / "ecommerce.json").read_text(encoding="utf-8"))
    return Architecture.from_dict(data)


def load_sample_description() -> str:
    return (EXAMPLES_DIR / "ecommerce_description.txt").read_text(encoding="utf-8").strip()
