import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import check_cinel02_baseline_attribution as check


def _report(*, primary=10.0, be=2.0, total=20.0, topas_total=18.0, package="pkg", rate="rate"):
    categories = []
    for label, value, topas in (("primary_C", primary, 9.0),
                                ("secondary_C", 1.0, 0.9),
                                ("Z5_B", 0.5, 0.45),
                                ("Z4_Be", be, 1.8),
                                ("Z3_Li", 0.7, 0.63),
                                ("Z2_He", 1.5, 1.35),
                                ("Z1_H", 2.2, 1.98)):
        categories.append({
            "category": label,
            "gpu_integral_Gy": value,
            "topas_integral_Gy": topas,
        })
    return {
        "histories": 100,
        "energy_MeV_per_u": 200,
        "grid": {"shape": [2, 2, 2]},
        "scorer": "3D DoseToMedium; IDD is X/Y sum",
        "inputs": [
            {"path": f"/data/{package}.cinpkg", "sha256": package},
            {"path": f"/data/{rate}.csv", "sha256": rate},
            {"path": "/data/topas_total.bin", "sha256": "topas"},
        ],
        "categories": categories,
        "charged_total": {
            "category": "charged_total",
            "gpu_integral_Gy": total,
            "topas_integral_Gy": topas_total,
        },
    }


def test_compatibility_only_changes_allowed_categories():
    result = check.check(_report(), _report(be=0.0, total=18.0, topas_total=18.0))
    assert result["status"] == "pass"
    assert result["stable_category_failures"] == []
    assert any(row["category"] == "Z4_Be" and row["allowed_to_change"] for row in result["rows"])


def test_primary_change_fails_attribution_sanity_check():
    result = check.check(_report(), _report(primary=10.2))
    assert result["status"] == "fail"
    assert result["stable_category_failures"] == ["primary_C"]


def test_provenance_mismatch_is_reported_even_when_dose_is_stable():
    result = check.check(_report(), _report(package="other"))
    assert result["status"] == "fail"
    assert any("package input hash mismatch" in error for error in result["errors"])
