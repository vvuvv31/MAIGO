#!/usr/bin/env python3
"""Unit tests for Schneider mass-SP factors (real schneider_hu entry points)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schneider_hu import load_schneider_table, mass_sp_factor_from_weights


def main() -> int:
    table_path = Path("ct/HUtoMaterialSchneider.txt")
    if not table_path.is_file():
        table_path = Path("validation/topas/HUtoMaterialSchneider.txt")
    assert table_path.is_file(), f"missing Schneider table {table_path}"

    table = load_schneider_table(table_path)
    assert table.n_material_sections >= 2
    assert table.mass_sp_factor.size == table.n_material_sections
    assert table.mass_sp_I_eV.size == table.n_material_sections

    # Water composition self-check.
    if table.elements:
        w = np.zeros(len(table.elements), dtype=np.float64)
        for i, name in enumerate(table.elements):
            if name == "Hydrogen":
                w[i] = 0.111894
            elif name == "Oxygen":
                w[i] = 0.888106
        f_water = mass_sp_factor_from_weights(w, table.elements)
        assert abs(f_water - 1.0) < 1.0e-6, f_water

    # Soft tissue sections near water; bone sections slightly below 1.
    soft = float(table.mass_sp_factor[min(7, table.n_material_sections - 1)])
    bone = float(table.mass_sp_factor[min(15, table.n_material_sections - 1)])
    soft_I = float(table.mass_sp_I_eV[min(7, table.n_material_sections - 1)])
    bone_I = float(table.mass_sp_I_eV[min(15, table.n_material_sections - 1)])
    assert 0.97 < soft < 1.03, soft
    assert 0.90 < bone < 1.0, bone
    assert 60.0 < soft_I < 90.0, soft_I
    assert 70.0 < bone_I < 150.0, bone_I
    assert bone_I > soft_I  # bone mean I typically above soft tissue

    dens, sec = __import__("schneider_hu", fromlist=["hu_to_density_material"]).hu_to_density_material(
        np.array([0.0, 854.0], dtype=np.float32), table, use_section_id=True
    )
    assert dens[0] > 0.9 and dens[0] < 1.1
    assert dens[1] > 1.4
    assert int(sec[1]) >= 8  # bone-ish section

    print(
        "test_schneider_mass_sp OK:",
        f"n_sec={table.n_material_sections}",
        f"soft_factor={soft:.4f}",
        f"bone_factor={bone:.4f}",
        f"soft_I={soft_I:.1f}",
        f"bone_I={bone_I:.1f}",
        f"HU0_rho={dens[0]:.4f}",
        f"HU854_rho={dens[1]:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
