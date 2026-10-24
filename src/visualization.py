"""Matplotlib rendering of the dependency graph (layered: dependents on top, foundations at the bottom)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from .graph_engine import SimulationResult, build_graph  # noqa: E402
from .models import Architecture  # noqa: E402

STATUS_COLORS = {
    "initial": "#b71c1c", "failed": "#ef6c00", "degraded": "#f9a825",
    "surviving": "#2e7d32", "neutral": "#546e7a",
}
PROVENANCE_COLORS = {"user_provided": "#1565c0", "model_inferred": "#8e24aa", "sample_data": "#757575"}


def layered_positions(g: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Layer = longest dependency chain below the node (cycle-safe via condensation)."""
    cond = nx.condensation(g)
    layer: dict[int, int] = {}
    for n in reversed(list(nx.topological_sort(cond))):  # dependencies first
        layer[n] = max((layer[s] + 1 for s in cond.successors(n)), default=0)
    rows: dict[int, list[str]] = {}
    for n in g.nodes:
        rows.setdefault(layer[cond.graph["mapping"][n]], []).append(n)
    pos = {}
    for lvl, nodes in rows.items():
        nodes.sort()
        for i, n in enumerate(nodes):
            pos[n] = ((i - (len(nodes) - 1) / 2) * 2.4, lvl * 1.8)
    return pos


def draw_graph(arch: Architecture, result: SimulationResult | None = None, title: str | None = None):
    g = build_graph(arch)
    pos = layered_positions(g)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = []
    for n in g.nodes:
        if result is None:
            colors.append(STATUS_COLORS["neutral"])
        elif n in result.initial:
            colors.append(STATUS_COLORS["initial"])
        elif n in result.failed:
            colors.append(STATUS_COLORS["failed"])
        elif n in result.degraded:
            colors.append(STATUS_COLORS["degraded"])
        else:
            colors.append(STATUS_COLORS["surviving"])
    for crit, style in (("required", "solid"), ("optional", "dashed")):
        edges = [(u, v) for u, v, a in g.edges(data=True) if a["criticality"] == crit]
        nx.draw_networkx_edges(
            g, pos, edgelist=edges, ax=ax, style=style, arrows=True, arrowsize=16, width=1.6,
            edge_color=[PROVENANCE_COLORS[g.edges[e]["provenance"]] for e in edges],
            node_size=3200, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=3200, edgecolors="#263238", linewidths=1.2)
    nx.draw_networkx_labels(g, pos, labels={n: arch.name_of(n).replace(" ", "\n", 1) for n in g.nodes},
                            font_size=7.5, font_color="white", font_weight="bold", ax=ax)
    ax.set_title(title or "Dependency graph  (arrow: A -> B means A depends on B)", fontsize=11)
    ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout()
    return fig
