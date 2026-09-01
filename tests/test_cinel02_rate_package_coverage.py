import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import census_cinel02_rate_package_coverage as census


def test_merge_replay_windows_and_support_membership():
    intervals = census._merge_intervals([(1.0 - 0.51, 1.0 + 0.51), (1.8 - 0.51, 1.8 + 0.51), (5.0 - 0.51, 5.0 + 0.51)])
    assert intervals == [[0.49, 2.31], [4.49, 5.51]]
    assert census._inside(2.0, intervals)
    assert not census._inside(3.0, intervals)


def test_nonzero_rate_sample_intervals_ignore_zero_samples():
    intervals = census._sample_support_intervals([(0.5, 0.0), (1.5, 2.0), (2.5, 3.0), (3.5, 0.0), (4.5, 4.0)])
    assert intervals == [[1.0, 3.0], [4.0, 5.0]]
