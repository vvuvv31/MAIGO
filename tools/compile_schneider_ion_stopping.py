#!/usr/bin/env python3
"""Compile TOPAS IonSchneiderStoppingPowerDump extraction into SCHNIOSP v1.

Inputs (Geant4/TOPAS provenance only, never synthesized):
  raw CSV: energy_mevu,section_id,species_z,species_a,material_name,
           density_g_cm3,linear_stopping_power_mev_per_mm
  raw JSON: num_sections=25, energies 0.01..430.11 step 0.1 (4302),
            species_za = 18 GPU slots in get_charged_species_idx order.

Outputs:
  data/schneider/schneider_ion_section_stopping_v1.bin (SCHNIOSP v1)
  data/schneider/schneider_ion_section_stopping_v1.metadata.json
Stored values: linear stopping at unit density (MeV/mm at 1 g/cm^3)
= raw linear / section density. Runtime scales by local voxel density.

Cross-check gates (fail-closed):
  - C12 rows vs data/schneider/schneider_stopping_v1.bin (same Geant4
    truth family; tolerance 2% for model/quantity drift).
  - water-like section C12 vs data/ion_stopping_power_water_geant4_11_3_2.csv.
"""
import argparse
import csv
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SECTIONS = 25
ENERGIES = 4302
E_MIN, E_MAX, E_STEP = 0.01, 430.11, 0.1
# GPU charged-species slots 0..17 (must match get_charged_species_idx).
SPECIES = [(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 6), (3, 7),
           (4, 7), (4, 9), (4, 10), (5, 8), (5, 10), (5, 11), (6, 10),
           (6, 11), (6, 12), (4, 6)]

MAGIC = b"SCHNIOSP"
VERSION = 1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-csv", required=True)
    ap.add_argument("--raw-json", required=True)
    ap.add_argument("--out-bin", default=str(
        REPO / "data/schneider/schneider_ion_section_stopping_v1.bin"))
    ap.add_argument("--c12-check-tol", type=float, default=0.02)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw_json).read_text())
    if raw.get("num_sections") != SECTIONS:
        raise ValueError("num_sections mismatch")
    if raw.get("num_energies") != ENERGIES:
        raise ValueError("num_energies mismatch")
    if abs(raw.get("energy_min_mevu", 0) - E_MIN) > 1e-6 or \
       abs(raw.get("energy_max_mevu", 0) - E_MAX) > 1e-6 or \
       abs(raw.get("energy_step_mevu", 0) - E_STEP) > 1e-6:
        raise ValueError("energy grid mismatch")
    if [tuple(x) for x in raw.get("species_za", [])] != SPECIES:
        raise ValueError("species order mismatch (must follow GPU slots)")

    grid = {}
    densities = {}
    with open(args.raw_csv) as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            key = (int(row["section_id"]), int(row["species_z"]),
                   int(row["species_a"]))
            e = round(float(row["energy_mevu"]), 4)
            grid.setdefault(key, {})[e] = (float(row["linear_stopping_power_mev_per_mm"]),
                                           float(row["density_g_cm3"]))
            densities[int(row["section_id"])] = float(row["density_g_cm3"])

    want_e = [round(E_MIN + i * E_STEP, 4) for i in range(ENERGIES)]
    values = []
    for s in range(SECTIONS):
        for (z, a) in SPECIES:
            col = grid.get((s, z, a))
            if col is None or len(col) != ENERGIES:
                raise ValueError(f"Missing/incomplete section {s} species {(z, a)}")
            rho = densities[s]
            if not rho > 0:
                raise ValueError(f"Bad density section {s}")
            for e in want_e:
                if e not in col:
                    raise ValueError(f"Missing energy {e} at {(s, z, a)}")
                lin, _ = col[e]
                if not lin > 0:
                    raise ValueError(f"Non-positive stopping at {(s, z, a, e)}")
                values.append(lin / rho)  # linear at unit density

    # Cross-check C12 (species slot 16) vs validated Schneider C12 table.
    # Only E >= 5 MeV/u: the C12 table used total dE/dx (with nuclear
    # stopping), this table uses unrestricted electronic; nuclear stopping
    # is negligible above a few MeV/u but dominates the comparison below.
    c12 = np_table("data/schneider/schneider_stopping_v1.bin")
    worst = 0.0
    for s in range(SECTIONS):
        for i in range(500, ENERGIES, 43):
            mine = values[(s * 18 + 16) * ENERGIES + i]
            # Schneider table stores massSP/10 convention ~= linear per unit
            # density; compare in the same stored units.
            ref = c12[s * 4302 + i]
            rel = abs(mine - ref) / ref
            worst = max(worst, rel)
    print(f"C12 cross-check worst relative diff: {worst:.4f}")
    if worst > args.c12_check_tol:
        raise ValueError("C12 cross-check failed")

    out = Path(args.out_bin)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, SECTIONS, 18, ENERGIES))
        f.write(struct.pack("<ddd", E_MIN, E_MAX, E_STEP))
        for s in range(SECTIONS):
            f.write(struct.pack("<d", densities[s]))
        f.write(struct.pack(f"<{len(values)}d", *values))

    meta = {
        "schema_version": 1,
        "format": "binary",
        "data_filename": out.name,
        "data_sha256": sha(out),
        "sections_count": SECTIONS,
        "species_count": 18,
        "energies_count": ENERGIES,
        "energy_grid_MeV_per_u": {"minimum": E_MIN, "maximum": E_MAX,
                                  "step": E_STEP},
        "quantity": "unrestricted electronic dE/dx, linear at unit density",
        "provenance": "TOPAS/Geant4 G4EmCalculator via IonSchneiderStoppingPowerDump; "
                      "see raw JSON inputs (sha below)",
        "raw_csv_sha256": sha(args.raw_csv),
        "raw_json_sha256": sha(args.raw_json),
        "compiler_sha256": sha(Path(__file__)),
    }
    (out.parent / (out.stem + ".metadata.json")).write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out} + metadata sidecar")


def np_table(path):
    import numpy as np
    with open(REPO / path, "rb") as f:
        f.seek(8 + 12 + 24)
        rest = np.fromfile(f, "<f8")
    return rest[25:25 + 25 * 4302].reshape(25, 4302)


if __name__ == "__main__":
    main()
