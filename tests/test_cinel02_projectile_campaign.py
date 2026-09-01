import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import audit_cinel02_projectile_campaign as audit


def test_summary_identifies_missing_be6_campaign(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "projectile_z,projectile_a,source_energy_MeV_per_u,histories,version,completed,raw_workers,interactions,products,raw_bytes,interactions_per_history\n"
        "4,7,200,50000,v4,1,60,10,20,100,0.0002\n",
        encoding="utf-8",
    )
    report = audit.audit(summary)
    assert report["be6_assessment"]["source_campaign_present"] is False
    assert report["summary"]["projectile_identity_count"] == 1
