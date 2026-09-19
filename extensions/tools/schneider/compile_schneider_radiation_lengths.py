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

import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED_TRUTH_SHA256 = "d8a5c771b4c74949991a4e089846d1e4a92022c056ca60eebf8e9218b8d93b6b"
EXPECTED_SECTIONS = 25

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    repo = Path(os.environ.get(
        "MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3])).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-json", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "data" / "schneider")
    parser.add_argument("--expected-truth-sha256",
                        default=EXPECTED_TRUTH_SHA256,
                        help="Set to the source hash for a new package version")
    args = parser.parse_args()
    truth_json = args.truth_json.resolve()
    out_dir = args.output_dir.resolve()
    out_json = out_dir / "schneider_radiation_lengths.json"
    out_meta = out_dir / "schneider_radiation_lengths.metadata.json"

    print("=== Step 15: Schneider Radiation Length Compiler ===")
    if not truth_json.is_file():
        raise FileNotFoundError(f"Missing TOPAS material truth file: {truth_json}")

    actual_truth_sha = sha256_file(truth_json)
    if actual_truth_sha != args.expected_truth_sha256:
        raise ValueError(
            f"Truth file SHA256 mismatch! Got {actual_truth_sha}, "
            f"expected {args.expected_truth_sha256}"
        )
    print(f"Verified source truth SHA256: {actual_truth_sha}")

    with open(truth_json, "r", encoding="utf-8") as f:
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
        "source_truth_sha256": actual_truth_sha,
        "sections": compiled_sections
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(product, f, indent=2)
    print(f"Wrote canonical JSON: {out_json}")

    out_json_sha = sha256_file(out_json)
    metadata = {
        "schema_version": 1,
        "data_filename": "schneider_radiation_lengths.json",
        "data_sha256": out_json_sha,
        "source_truth_sha256": actual_truth_sha,
        "sections_count": EXPECTED_SECTIONS,
        "values_g_per_cm2": values_g_cm2
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {out_meta}")
    print("Compilation completed successfully!")

if __name__ == "__main__":
    main()
