import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "startup" / "package_tools"))
from cinel02_coverage_grid import bin_index, hybrid_edges


def test_hybrid_grid_segments_and_boundaries():
    edges = hybrid_edges(400.0)
    assert edges[:11] == [float(value) for value in range(11)]
    assert edges[11:31] == [float(value) for value in range(12, 51, 2)]
    assert edges[31] == 52.5
    assert bin_index(9.999, edges) == 9
    assert bin_index(10.0, edges) == 10
    assert bin_index(49.999, edges) == 29
    assert bin_index(50.0, edges) == 30
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))
