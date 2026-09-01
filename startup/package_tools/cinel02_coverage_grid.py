"""Authoritative hybrid-log energy grid for CINEL02 coverage qualification."""

from __future__ import annotations

import bisect
import math


def hybrid_edges(maximum_MeV_per_u: float = 2000.0) -> list[float]:
    if maximum_MeV_per_u <= 50.0:
        maximum_MeV_per_u = 50.0
    edges = [float(value) for value in range(0, 11)]
    edges.extend(float(value) for value in range(12, 51, 2))
    value = 50.0
    while value < maximum_MeV_per_u:
        # Exact geometric growth prevents arbitrary integer-boundary artifacts.
        value *= 1.05
        edges.append(value)
    return edges


def bin_index(energy_MeV_per_u: float, edges: list[float]) -> int:
    if not math.isfinite(energy_MeV_per_u) or energy_MeV_per_u < edges[0]:
        raise ValueError(f"energy outside hybrid grid: {energy_MeV_per_u}")
    index = bisect.bisect_right(edges, energy_MeV_per_u) - 1
    return min(index, len(edges) - 2)
