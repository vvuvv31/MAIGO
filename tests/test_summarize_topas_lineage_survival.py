from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "startup" / "package_tools"))

from summarize_topas_lineage_survival import (
    _COLUMN_NAMES,
    summarize,
)


def test_isotope_generation_partition_and_header_units() -> None:
    with tempfile.TemporaryDirectory() as directory:
        stem = Path(directory) / "lineage"
        labels = list(_COLUMN_NAMES)
        header = [
            "Number of Original Histories: 3",
            "Number of Scored Entries: 3",
            "",
            "Columns of data are as follows:",
        ]
        header.extend(
            f"{index}: {label}{' [MeV]' if 'Kinetic Energy' in label else ''}"
            for index, label in enumerate(labels, start=1)
        )
        stem.with_suffix(".header").write_text("\n".join(header) + "\n")
        rows = [
            ["episode", 0, 0, 1, 10, 1, 3, 6, 0, 0, 4, 0, 0, 1, 100, 0, 0, 5, 0, 0, 0, 0, 0, 0, "CarbonInelastic", "stopped", "ionIoni"],
            ["episode", 0, 0, 1, 11, 1, 3, 6, 0, 0, 7, 1, 0, 1, 200, 150, 150, 10, 0, 0, 0, 0, 0, 0, "CarbonInelastic", "hadronic_interaction", "CarbonInelastic"],
            ["episode", 0, 0, 2, 12, 1, 4, 9, 1, 0, 3, 0, 1, 1, 300, 0, 0, 20, 0, 0, 0, 0, 0, 0, "CarbonInelastic", "event_end_censored", "none"],
        ]
        rows = [" ".join(map(str, row)) for row in rows]
        stem.with_suffix(".phsp").write_text("\n".join(rows) + "\n")
        report = summarize(stem)
        assert report["row_count"] == 3
        assert report["episode_count"] == 3
        by_key = {
            (row["atomic_number"], row["atomic_mass"], row["generation"]): row
            for row in report["isotope_generation"]
        }
        assert by_key[(3, 6, 0)]["episode_count"] == 2
        assert by_key[(3, 6, 0)]["reacted_count"] == 1
        assert by_key[(4, 9, 1)]["censored_count"] == 1


if __name__ == "__main__":
    test_isotope_generation_partition_and_header_units()
    print("test_isotope_generation_partition_and_header_units: PASS")
