#!/usr/bin/env python3
"""Schneider C12 elastic rate v1 compiler: grid [0.1, 460.1] step 0.5
(921 nodes), mirroring the SCHNRATE v3 layout with magic SCHNELXS v1.

Inputs (TOPAS/Geant4 provenance only):
- raw dump JSON from CarbonSchneiderElasticXsDump (25 sections,
  per-energy per-element mass_partial_per_mm_at_1g_cm3, fHadronElastic).

Rules (no package domains exist for elastic):
- XS==0 exact -> 0, kept; raw positives kept everywhere.
- per-target domain = [min, max] node with XS>0 (has_support=1);
  all-zero target -> has_support=0, [0, 0].
- totals = raw sums (bitwise closure with the loader check).

Layout: SCHNELXS version 1: header, 25 sections x 13 targets x 921
partials, 25x921 totals, 13 domain entries {double emin, double emax,
uint8 has + 7 pad}.

Outputs:
  data/schneider/schneider_elastic_rates_v1.bin (+ .metadata.json)
  data/schneider/c12_schneider_elastic_mass_xs_v1.csv (audit artifact)
"""
import argparse
import datetime
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EMIN, EMAX, ESTEP, NE = 0.1, 460.1, 0.5, 921
CANON_Z = [1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22]
MAGIC = b"SCHNELXS"
# binary_version 3 = shared v3 layout schema (921 + domains);
# magic distinguishes the elastic family. Data release v1.
VERSION = 3


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-json", required=True)
    ap.add_argument("--out-bin", default=str(
        REPO / "data/schneider/schneider_elastic_rates_v1.bin"))
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--topas-version", default="4.2.p3")
    ap.add_argument("--geant4-version", default="11.03.p02")
    args = ap.parse_args()

    d = json.load(open(args.raw_json))
    if (d.get("projectile_z"), d.get("projectile_a")) != (6, 12):
        raise ValueError("elastic dump must use C12 projectile")
    prov = d.get("process_provenance", {})
    if prov.get("process_sub_type_name") != "fHadronElastic":
        raise ValueError("dump must record fHadronElastic provenance")
    secs = d["sections"]
    if len(secs) != 25 or len(secs[0]["grid"]) != NE:
        raise ValueError("grid dimension mismatch")

    partial = {}
    for s, sec in enumerate(secs):
        for j, pt in enumerate(sec["grid"]):
            e = EMIN + j * ESTEP
            if abs(pt["energy_mevu"] - e) > 1e-9:
                raise ValueError(f"energy node mismatch at {(s, j)}")
            for el in pt["elements"]:
                tz = el["target_z"]
                xs = el["mass_partial_per_mm_at_1g_cm3"]
                if not xs >= 0.0:
                    raise ValueError(f"bad xs at {(s, tz, j)}")
                partial[(s, tz, j)] = xs if xs > 0.0 else 0.0
    total = {}
    for s in range(25):
        for j in range(NE):
            # Explicit sequential accumulation: must be bitwise-identical
            # to the C++ loader closure check (Python 3.12 builtin sum()
            # uses Neumaier compensation and rounds differently).
            acc = 0.0
            for tz in CANON_Z:
                acc += partial[(s, tz, j)]
            total[(s, j)] = acc
    domains = {}
    for tz in CANON_Z:
        lo, hi, anypos = None, None, False
        for s in range(25):
            for j in range(NE):
                if partial[(s, tz, j)] > 0.0:
                    e = EMIN + j * ESTEP
                    lo = e if lo is None else min(lo, e)
                    hi = e if hi is None else max(hi, e)
                    anypos = True
        domains[tz] = (lo or 0.0, hi or 0.0, anypos)

    out_bin = Path(args.out_bin)
    csv_path = Path(args.out_csv) if args.out_csv else out_bin.parent / "c12_schneider_elastic_mass_xs_v1.csv"
    tmpdir = Path(tempfile.mkdtemp(prefix="elratev1_", dir=REPO / "data/schneider"))
    tmp_csv = tmpdir / csv_path.name
    with open(tmp_csv, "w") as f:
        f.write("energy_MeV_per_u," + ",".join(
            f"section_{s:02d}_mass_xs_per_mm_at_1g_cm3" for s in range(25)) + "\n")
        for j in range(NE):
            e = EMIN + j * ESTEP
            f.write(f"{e:.6f}," + ",".join(f"{total[(s, j)]:.12e}" for s in range(25)) + "\n")
    tmp = tmpdir / out_bin.name
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<III", 25, 13, NE))
        f.write(struct.pack("<ddd", EMIN, EMAX, ESTEP))
        f.write(struct.pack("<13i", *CANON_Z))
        for s in range(25):
            for tz in CANON_Z:
                for j in range(NE):
                    f.write(struct.pack("<d", partial[(s, tz, j)]))
        for s in range(25):
            for j in range(NE):
                f.write(struct.pack("<d", total[(s, j)]))
        for tz in CANON_Z:
            flo, fhi, has = domains[tz][0], domains[tz][1], domains[tz][2]
            f.write(struct.pack("<dd", flo, fhi))
            f.write(struct.pack("<B", 1 if has else 0))
            f.write(b"\x00" * 7)
    bin_sha = sha256_file(tmp)
    csv_sha = sha256_file(tmp_csv)
    meta = {
        "schema_version": 1, "format": "SCHNELXS", "binary_version": VERSION,
        "data_filename": out_bin.name, "data_sha256": bin_sha,
        "file_size_bytes": tmp.stat().st_size, "binary_magic": "SCHNELXS",
        "target_order": CANON_Z,
        "energy_grid": {"emin": EMIN, "emax": EMAX, "step": ESTEP, "count": NE},
        "channel_domains": [
            {"target_z": tz, "energy_min_mevu": domains[tz][0],
             "energy_max_mevu": domains[tz][1],
             "has_support": bool(domains[tz][2])} for tz in CANON_Z],
        "companion_csv": {"filename": csv_path.name, "sha256": csv_sha,
                          "note": "raw totals audit artifact; NOT loaded by runtime"},
        "topas_version": args.topas_version, "geant4_version": args.geant4_version,
        "physics_list": "FTFP_INCLXX",
        "schneider_source_file": "data/HUtoMaterialSchneider.txt",
        "schneider_sha256": sha256_file(REPO / "data/HUtoMaterialSchneider.txt"),
        "zeroing_policy": "zero iff extracted XS==0; raw positives kept "
                          "(no package domains exist for elastic)",
        "units": "mm, MeV, ns, g/cm^3",
        "generation_timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "compiler_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "compiler_command": "python3 tools/compile_schneider_elastic_rates_v1.py " + " ".join(sys.argv[1:]),
    }
    import shutil
    shutil.move(str(tmp), str(out_bin))
    shutil.move(str(tmp_csv), str(csv_path))
    (out_bin.parent / (out_bin.stem + ".metadata.json")).write_text(
        json.dumps(meta, indent=1))
    shutil.rmtree(tmpdir)
    print(f"compiled {out_bin} ({out_bin.stat().st_size/1e6:.1f} MB); csv {csv_path}")


if __name__ == "__main__":
    main()
