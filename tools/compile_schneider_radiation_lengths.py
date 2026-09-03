#!/usr/bin/env python3
"""
tools/compile_schneider_radiation_lengths.py

Extracts, compiles, and cryptographically anchors the 25-section Schneider
radiation lengths from the validated TOPAS/Geant4 material truth product
(anchored in plan/evidence-step03.sha256).

Generates:
  - data/schneider/schneider_radiation_lengths.json
  - data/schneider/schneider_radiation_lengths.metadata.json
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path("/mnt/sdb/wuwei/MAIGO")
TRUTH_JSON = Path("/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03/topas-schneider-materials.json")
ANCHOR_FILE = REPO_ROOT / "plan/evidence-step03.sha256"
OUT_JSON = REPO_ROOT / "data/schneider/schneider_radiation_lengths.json"
OUT_META = REPO_ROOT / "data/schneider/schneider_radiation_lengths.metadata.json"

EXPECTED_TRUTH_SHA256 = "d8a5c771b4c74949991a4e089846d1e4a92022c056ca60eebf8e9218b8d93b6b"
EXPECTED_SECTIONS = 25

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Step 15: Schneider Radiation Length Compiler ===")
    if not TRUTH_JSON.is_file():
        raise FileNotFoundError(f"Missing TOPAS material truth file: {TRUTH_JSON}")
    if not ANCHOR_FILE.is_file():
        raise FileNotFoundError(f"Missing anchor file: {ANCHOR_FILE}")

    actual_truth_sha = sha256_file(TRUTH_JSON)
    if actual_truth_sha != EXPECTED_TRUTH_SHA256:
        raise ValueError(
            f"Truth file SHA256 mismatch! Got {actual_truth_sha}, expected {EXPECTED_TRUTH_SHA256}"
        )
    print(f"Verified source truth SHA256: {actual_truth_sha} (matches anchor)")

    with open(TRUTH_JSON, "r", encoding="utf-8") as f:
        truth_data = json.load(f)

    sections = truth_data.get("sections", [])
    if len(sections) != EXPECTED_SECTIONS:
        raise ValueError(f"Expected {EXPECTED_SECTIONS} sections, got {len(sections)}")

    compiled_sections = []
    values_g_cm2 = []

    for idx, sec in enumerate(sections):
        sec_id = sec.get("section_id")
        if sec_id != idx:
            raise ValueError(f"Section ordering mismatch: expected {idx}, got {sec_id}")

        hu = sec["representative_HU"]
        mat_name = sec["material_name"]
        rho = float(sec["density_g_cm3"])
        rad_len_mm = float(sec["radiation_length_mm"])

        rad_len_g_cm2 = (rad_len_mm * 0.1) * rho
        values_g_cm2.append(rad_len_g_cm2)

        compiled_sections.append({
            "section_id": sec_id,
            "representative_hu": hu,
            "material_name": mat_name,
            "density_g_cm3": rho,
            "radiation_length_mm": rad_len_mm,
            "radiation_length_g_per_cm2": rad_len_g_cm2
        })

    product = {
        "schema_version": 1,
        "description": "Exact 25-section Schneider radiation lengths derived from TOPAS/Geant4 11.03.p02",
        "number_of_sections": EXPECTED_SECTIONS,
        "source_truth_file": "topas-schneider-materials.json",
        "source_truth_sha256": EXPECTED_TRUTH_SHA256,
        "sections": compiled_sections
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(product, f, indent=2)
    print(f"Wrote canonical JSON: {OUT_JSON}")

    out_json_sha = sha256_file(OUT_JSON)
    metadata = {
        "schema_version": 1,
        "data_filename": "schneider_radiation_lengths.json",
        "data_sha256": out_json_sha,
        "source_truth_sha256": EXPECTED_TRUTH_SHA256,
        "sections_count": EXPECTED_SECTIONS,
        "values_g_per_cm2": values_g_cm2
    }

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {OUT_META}")
    print("Compilation completed successfully!")

if __name__ == "__main__":
    main()
