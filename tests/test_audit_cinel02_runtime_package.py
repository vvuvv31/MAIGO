import importlib.util
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
MODULE_PATH = TOOLS / "audit_cinel02_runtime_package.py"
SPEC = importlib.util.spec_from_file_location("audit_cinel02_runtime_package", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_parent_outcome_summary_preserves_cell_occupancy():
    rows = []
    for expected_continue, expected_kill in ((1.0, 3.0), (2.0, 2.0)):
        rows.append({
            "species": 8,
            "target": 0,
            "reaction_generation": 1,
            "expected_valid_product_counts": [0] * 18,
            "expected_valid_product_kinetic_MeV": [0.0] * 18,
            "expected_valid_product_count_stddev": [0.0] * 18,
            "expected_valid_parent_continue_count": expected_continue,
            "expected_valid_parent_kill_count": expected_kill,
            "support_qualified": True,
            "valid_count": int(expected_continue + expected_kill),
            "target": 0,
            "energy_bin": 10,
        })
    ledger = {
        "cinel02_generated_transition_counts": [0] * (18 * 18),
        "cinel02_queued_transition_counts": [0] * (18 * 18),
        "cinel02_parent_outcome_counts": [0] * (18 * 2 * 3 * 2),
    }
    # 7Be (species 8), H (target 0), reaction generation 1.
    cell = (8 * 2 + 0) * 3 + 1
    ledger["cinel02_parent_outcome_counts"][cell * 2] = 3
    ledger["cinel02_parent_outcome_counts"][cell * 2 + 1] = 5
    summary = audit._summarise(rows, ledger)
    outcome = next(item for item in summary["parent_outcome_rows"]
                   if item["projectile"] == "7Be" and item["target"] == "H"
                   and item["reaction_generation"] == 1)
    assert outcome["actual_continue_count"] == 3
    assert outcome["actual_kill_count"] == 5
    assert outcome["expected_continue_count"] == 3.0
    assert outcome["expected_kill_count"] == 5.0
    assert outcome["actual_continue_fraction"] == 3 / 8
    assert outcome["expected_continue_fraction"] == 3 / 8
