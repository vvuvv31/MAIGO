#!/usr/bin/env python3
"""Primary rate v2.1 compiler: C12 grid [0.1, 460.1] step 0.5 (921 nodes),
binary version 3 with per-target valid-domain block and masked totals.

Inputs:
- /mnt/sda/wuwei/secondary-rates-v2-1/raw/c12_dump.json (deterministic XS)
- primary v2.1 channels (authoritative support map: has_support + bounds)
- campaign_manifest_all.json + raw_validation_all.json (C12 null evidence)

Rules (same as secondary v3): XS==0 exact -> 0; E > emax -> 0 (package is
the support map); E < emin with XS>0 -> 0 ONLY with null evidence, else
NEED (kept raw, runtime-masked). Stored totals = masked-at-node sums in
ascending-target order (C++ loader requires bitwise equality). Node energy
for masking is EMIN + j*ESTEP (bitwise agreement with the loader).

Layout: SCHNRATE version 3: header, 25 sections x 13 targets x 921 partials,
25x921 totals, 13 domain entries {double emin, double emax, uint8 has + 7 pad}.

Outputs (new names; v1 untouched):
  data/schneider/c12_schneider_inelastic_mass_xs_v2_1.csv (+ .metadata.json)
  data/schneider/schneider_inelastic_rates_v2_1.bin (+ .metadata.json)
"""
import datetime
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/mnt/sdb/wuwei/MAIGO")
V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")
DUMP = Path("/mnt/sda/wuwei/secondary-rates-v2-1/raw/c12_dump.json")
EMIN, EMAX, ESTEP, NE = 0.1, 460.1, 0.5, 921
CANON_Z = [1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22]
CHANNELS = "cinel03_c12_targets_v2_1.channels.json"
DRY_RUN = "--dry-run" in sys.argv


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def c12_domains():
    ch = json.load(open(REPO / ("data/schneider/" + CHANNELS)))
    out = {}
    for c in ch["channels"]:
        assert (c["projectile_z"], c["projectile_a"]) == (6, 12)
        lo, hi = c["energy_min_MeV_per_u"], c["energy_max_MeV_per_u"]
        assert f32(lo) == lo and f32(hi) == hi, c["target_element_z"]
        out[c["target_element_z"]] = (lo, hi)
    assert len(out) == 13, f"C12 needs 13 target domains, got {len(out)}"
    return out


def c12_nulls():
    val = json.load(open(V2 / "raw_validation_all.json"))
    man = json.load(open(V2 / "campaign_manifest_all.json"))
    by_task = {t["task_index"]: t for t in man["tasks"]}
    out = {}
    for r in val["results"]:
        if r["pass"]:
            continue
        t = by_task[r["task_index"]]
        if (t["projectile_Z"], t["projectile_A"]) != (6, 12):
            continue
        if not r["checks"].get("files_exist_nonempty"):
            out[t["target_Z"]] = max(out.get(t["target_Z"], -1.0),
                                     t["energy_MeV_per_u"])
    return out


def main():
    d = json.load(open(DUMP))
    assert (d["projectile_z"], d["projectile_a"]) == (6, 12)
    assert d["process_provenance"]["process_name"] == "ionInelastic"
    secs = d["sections"]
    assert len(secs) == 25 and len(secs[0]["grid"]) == NE
    doms = c12_domains()
    nulls = c12_nulls()
    print("C12 domains:", len(doms), "null targets:", len(nulls))

    partial = {}
    zeroed_zero = zeroed_below = zeroed_above = 0
    need = []
    for s, sec in enumerate(secs):
        for j, pt in enumerate(sec["grid"]):
            e = EMIN + j * ESTEP
            assert abs(pt["energy_mevu"] - e) < 1e-9, (s, j)
            for el in pt["elements"]:
                tz = el["target_z"]
                xs = el["mass_partial_per_mm_at_1g_cm3"]
                flo, fhi = doms[tz]
                keep = True
                if xs == 0.0:
                    keep = False
                    zeroed_zero += 1
                elif e > fhi + 1e-9:
                    keep = False
                    zeroed_above += 1
                elif e < flo - 1e-9:
                    if e <= nulls.get(tz, -1.0) + 1e-9:
                        keep = False
                        zeroed_below += 1
                    else:
                        need.append((tz, e, xs))
                if not keep:
                    xs = 0.0
                partial[(s, tz, j)] = xs
    total = {}
    for s in range(25):
        for j in range(NE):
            e = EMIN + j * ESTEP
            tot = 0.0
            for tz in CANON_Z:
                xs = partial[(s, tz, j)]
                flo, fhi = doms[tz]
                if e < flo or e > fhi:
                    xs = 0.0
                tot += xs
            total[(s, j)] = tot
    probes = []
    seen = {}
    for tz, e, xs in sorted(need):
        if e - seen.get(tz, -1e9) >= 1.9:
            seen[tz] = e
            probes.append({"target_Z": tz, "energy_MeV_per_u": round(e, 2),
                           "xs": xs})
    json.dump({"need_probes": probes},
              open(V2 / "step28" / "c12_need_probes.json", "w"), indent=1)
    print(f"C12 NEED: {len(probes)}; zeroed xs0={zeroed_zero} "
          f"below-null={zeroed_below} above-emax={zeroed_above}")
    if DRY_RUN:
        print("dry-run: binaries NOT written")
        return
    if probes:
        frozen = json.load(open(V2 / "step28" / "c12_need_frozen_v2_1.json"))["need_probes"]
        if probes == frozen and "--accept-residual-need" in sys.argv:
            print(f"accepting {len(probes)} frozen residual C12 NEED points")
        else:
            raise RuntimeError(
                f"refusing final compile with {len(probes)} unprobed C12 NEED points")

    csv_path = REPO / "data/schneider/c12_schneider_inelastic_mass_xs_v2_1.csv"
    out_bin = REPO / "data/schneider/schneider_inelastic_rates_v2_1.bin"
    tmpdir = Path(tempfile.mkdtemp(prefix="priratv3_", dir=REPO / "data/schneider"))
    tmp_csv = tmpdir / csv_path.name
    with open(tmp_csv, "w") as f:
        f.write("energy_MeV_per_u," + ",".join(
            f"section_{s:02d}_mass_xs_per_mm_at_1g_cm3" for s in range(25)) + "\n")
        for j in range(NE):
            e = EMIN + j * ESTEP
            f.write(f"{e:.6f}," + ",".join(f"{total[(s, j)]:.12e}" for s in range(25)) + "\n")
    tmp = tmpdir / out_bin.name
    with open(tmp, "wb") as f:
        f.write(b"SCHNRATE")
        f.write(struct.pack("<I", 3))  # version 3: domain block
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
            flo, fhi = doms[tz]
            f.write(struct.pack("<dd", flo, fhi))
            f.write(struct.pack("<B", 1))
            f.write(b"\x00" * 7)
    bin_sha = sha256_file(tmp)
    csv_sha = sha256_file(tmp_csv)
    meta = {
        "schema_version": 1, "format": "SCHNRATE", "binary_version": 3,
        "data_filename": out_bin.name, "data_sha256": bin_sha,
        "file_size_bytes": tmp.stat().st_size, "binary_magic": "SCHNRATE",
        "target_order": CANON_Z,
        "energy_grid": {"emin": EMIN, "emax": EMAX, "step": ESTEP, "count": NE},
        "channel_domains": [
            {"target_z": tz, "energy_min_mevu": doms[tz][0],
             "energy_max_mevu": doms[tz][1], "has_support": True}
            for tz in CANON_Z],
        "companion_csv": {"filename": csv_path.name, "sha256": csv_sha,
                          "note": "masked totals audit artifact; NOT loaded by v3 runtime"},
        "topas_version": "4.2.p3", "geant4_version": "11.03.p02",
        "physics_list": "FTFP_INCLXX",
        "schneider_source_file": "data/HUtoMaterialSchneider.txt",
        "schneider_sha256": sha256_file(REPO / "data/HUtoMaterialSchneider.txt"),
        "zeroing_policy": ("zero iff XS==0, E <= per-target max-null-beam, "
                           "or E > target emax; substantial-XS-without-nodes "
                           "NEVER zeroed; totals = masked-at-node sums"),
        "zeroed_xs0": zeroed_zero, "zeroed_below_null": zeroed_below,
        "zeroed_above_emax": zeroed_above,
        "units": "mm, MeV, ns, g/cm^3",
        "generation_timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "compiler_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "compiler_command": "python3 tools/compile_primary_rates_v2_1.py " + " ".join(sys.argv[1:]),
    }
    tmp.replace(out_bin)
    tmp_csv.replace(csv_path)
    (REPO / ("data/schneider/" + out_bin.stem + ".metadata.json")).write_text(
        json.dumps(meta, indent=1))
    import shutil
    shutil.rmtree(tmpdir)
    print(f"compiled {out_bin} ({out_bin.stat().st_size/1e6:.1f} MB); csv {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
