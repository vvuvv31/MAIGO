from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "startup" / "package_tools" / "extract_ct_cinel02_rates.py"
SPEC = importlib.util.spec_from_file_location("extract_ct_cinel02_rates", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_builtin_schneider_table() -> None:
    table = MODULE.parse_schneider(ROOT / "data" / "HUtoMaterialSchneider.txt")
    assert table.section_count == 25
    assert table.boundaries[0] == -1000
    assert table.boundaries[-1] == 2996
    assert table.elements[0:4] == ("Hydrogen", "Carbon", "Nitrogen", "Oxygen")
    assert all(abs(sum(row) - 1.0) < 2.0e-3 for row in table.fractions)


def test_render_uses_pure_h1_o16_and_excludes_be6() -> None:
    table = MODULE.parse_schneider(ROOT / "data" / "HUtoMaterialSchneider.txt")
    assert (4, 6) not in MODULE.TRANSPORTABLE_PROJECTILES
    assert (6, 12) in MODULE.TRANSPORTABLE_PROJECTILES
    config = ROOT / "tests" / ".tmp_ct_cinel02_config.txt"
    try:
        MODULE.render_config(
            table, 7, config, "rate", histories=1, threads=1, seed=1,
            energy_min=0.0, energy_width=1.0, energy_count=3,
            reference_density=1.0,
        )
        text = config.read_text(encoding="utf-8")
        assert '"CTHydrogenH1"' in text
        assert '"CTOxygenO16"' in text
        assert "MaterialSection = 7" in text
        assert "EnergyCount = 3" in text
    finally:
        config.unlink(missing_ok=True)
