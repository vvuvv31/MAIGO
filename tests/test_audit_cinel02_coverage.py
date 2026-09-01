import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "startup" / "package_tools"
sys.path.insert(0, str(TOOLS))

import audit_cinel02_coverage as coverage
import cinel02


def _record(event, energy, target=(1, 1), projectile=(6, 12), products=1):
    record = {name: 0 for name in cinel02._FIELD_NAMES}
    for name in ("material_name", "target_isotope_name", "process_name", "model_name"):
        record[name] = "synthetic"
    record.update({
        "run_id": 1, "event_id": event, "track_id": 1,
        "projectile_pdg": cinel02._ground_state_ion_pdg(*projectile),
        "projectile_z": projectile[0], "projectile_a": projectile[1],
        "projectile_charge": float(projectile[0]), "projectile_rest_mass": projectile[1] * 931.494,
        "collision_energy_MeV": energy * projectile[1], "collision_energy_MeV_per_u": energy,
        "collision_direction_z": 1.0, "track_weight": 1.0,
        "target_z": target[0], "target_a": target[1],
        "parent_pdg": cinel02._ground_state_ion_pdg(*projectile),
        "parent_z": projectile[0], "parent_a": projectile[1],
        "parent_charge": float(projectile[0]), "parent_rest_mass": projectile[1] * 931.494,
        "parent_direction_z": 1.0, "direct_product_count": products,
        "process_local_deposit_MeV": 2.0,
    })
    return record


def _proton(role=cinel02.PRODUCT_DIRECT_SECONDARY):
    return {
        "pdg": 2212, "z": 1, "a": 1, "charge": 1.0, "rest_mass": 938.272,
        "excitation": 0.0, "kinetic_energy_MeV": 10.0,
        "direction_x": 0.0, "direction_y": 0.0, "direction_z": 1.0,
        "local_direction_x": 0.0, "local_direction_y": 0.0, "local_direction_z": 1.0,
        "position_x_mm": 0.0, "position_y_mm": 0.0, "position_z_mm": 0.0,
        "creation_time_ns": 0.0, "weight": 1.0, "role": role,
    }


def _write_raw(path, records):
    body = b"".join(cinel02.pack_record(record, products) for record, products in records)
    product_count = sum(len(products) for _, products in records)
    size = cinel02.RAW_HEADER_SIZE + len(body)
    header = cinel02.RAW_HEADER.pack(cinel02.RAW_MAGIC, cinel02.RAW_VERSION,
                                    cinel02.RAW_HEADER_SIZE, 0x01020304, 0,
                                    len(records), product_count, size, 0, 0)
    path.write_bytes(header + body)


def test_sparse_missing_target_and_energy_gap(tmp_path):
    raw = tmp_path / "synthetic.cinel02"
    _write_raw(raw, [
        (_record(1, 1.2), [_proton()]),
        (_record(2, 3.2), [_proton()]),
    ])
    report = coverage.audit([raw], campaign_uuid="synthetic")
    assert report["qualified"] is False
    assert report["summary"]["cell_count"] == 2
    assert report["summary"]["unqualified_cell_count"] == 2
    assert any(gap["kind"] == "missing_target" and gap["target_z"] == 8
               for gap in report["energy_gaps"])
    assert any(gap["kind"] == "energy_gap" and gap["energy_bin_id"] == 2
               for gap in report["energy_gaps"])


def test_qualified_cells_with_h_and_o(tmp_path):
    raw = tmp_path / "qualified.cinel02"
    records = []
    for target in ((1, 1), (8, 16)):
        for event in range(1000):
            records.append((_record(event + 1, 5.2, target=target), [_proton()]))
    _write_raw(raw, records)
    report = coverage.audit([raw], campaign_uuid="synthetic")
    assert report["qualified"] is True
    assert report["summary"]["qualified_cell_count"] == 2
    assert report["cells"][0]["species_yield_per_event"]["Z1A1"] == 1.0
