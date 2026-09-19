#!/usr/bin/env python3
"""Rate v2.1 compiler, binary version 3: grid [0.1, 460.1] step 0.5 (921
nodes), 14 projectiles (13 legacy + C12 secondary), per-channel valid
domains, masked totals.

Inputs:
- /mnt/sda/wuwei/secondary-rates-v2-1/raw/<pid>_dump.json (13 + c12)
- 14p package channels (authoritative support map: has_support + bounds)
- campaign_manifest_all.json + raw_validation_all.json (null evidence)

Rules (documented, evidence-backed, never silent):
- XS(E) == 0 exactly (deterministic model threshold) -> rate node 0.
- E > channel emax -> rate node 0 (package is the support map; no nodes
  exist above emax by construction). The device/host mask is authoritative
  at query time; node zeroing only shapes near-boundary interpolation.
- E < channel emin with XS(E) > 0 -> rate node 0 ONLY with dynamic-null
  evidence (campaign beam >= E with zero collisions AND zero captures);
  otherwise NEED-NODES (kept raw, probe first, never silent-zero).
- Stored totals = masked-at-node sums in ascending-target order (the C++
  loader re-sums identically and requires bitwise equality).
- Node energy for masking is grid-index arithmetic (EMIN + j*ESTEP), never
  the dump float (bitwise agreement with the C++ loader).

Layout: SCHN2RAT version 3:
  header, 14 projectile keys, 13 canonical targets,
  partials [14][25][13][921] doubles, totals [14][25][921] doubles,
  domain block [14][13] x {double emin, double emax, uint8 has + 7 pad}.

Outputs (new names; v1 untouched):
  data/schneider/secondary_inelastic_rates_v2_1.bin (.metadata.json)
"""
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3]))
WORK = Path(os.environ.get("MAIGO_PACKAGE_WORK_ROOT", REPO / "extensions" / "work"))
DUMP_RAW = Path(os.environ.get("MAIGO_RATE_DUMP_ROOT", WORK / "secondary-rates-v2-1" / "raw"))
V2 = Path(os.environ.get("MAIGO_V2_CAMPAIGN_ROOT", WORK / "schneider-secondary-v2"))
EMIN, EMAX, ESTEP, NE = 0.1, 460.1, 0.5, 921
PID_ORDER = ["b11", "b10", "be9", "be7", "be10", "li7", "li6",
             "he4", "he3", "h1", "h2", "h3", "c11", "c12"]
PROJ_ZA = {"b11": (5, 11), "b10": (5, 10), "be9": (4, 9), "be7": (4, 7),
           "be10": (4, 10), "li7": (3, 7), "li6": (3, 6), "he4": (2, 4),
           "he3": (2, 3), "h1": (1, 1), "h2": (1, 2), "h3": (1, 3),
           "c11": (6, 11), "c12": (6, 12)}
CANON_Z = [1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


CHANNELS_NAME = sys.argv[sys.argv.index("--channels") + 1] \
    if "--channels" in sys.argv else "cinel03_secondary_targets_v2_1_14p.channels.json"
DRY_RUN = "--dry-run" in sys.argv


def package_domains():
    """(emin, emax) per (pz,pa,tz) from a package channels.json. All 14x13
    channels must exist (asserted). Bounds must be float32-exact (device
    mask/package-node agreement)."""
    ch = json.load(open(REPO / ("data/schneider/" + CHANNELS_NAME)))
    out = {}
    for c in ch["channels"]:
        key = (c["projectile_z"], c["projectile_a"], c["target_element_z"])
        lo, hi = c["energy_min_MeV_per_u"], c["energy_max_MeV_per_u"]
        assert f32(lo) == lo, f"emin not f32-exact for {key}"
        assert f32(hi) == hi, f"emax not f32-exact for {key}"
        out[key] = (lo, hi)
    assert len(out) == 14 * 13, f"14p support map requires 182 channels, got {len(out)}"
    return out


def null_evidence():
    """Per-channel max null beam energy: campaigns with zero collisions AND
    zero captured events (dynamic Geant4 ground truth). Rate is zeroed at
    grid E <= max-null-beam (never by proximity radius)."""
    val = json.load(open(V2 / "raw_validation_all.json"))
    man = json.load(open(V2 / "campaign_manifest_all.json"))
    by_task = {t["task_index"]: t for t in man["tasks"]}
    maxnull = {}
    for r in val["results"]:
        if r["pass"]:
            continue
        t = by_task[r["task_index"]]
        if not r["checks"].get("files_exist_nonempty"):
            key = (t["projectile_Z"], t["projectile_A"], t["target_Z"])
            maxnull[key] = max(maxnull.get(key, -1.0), t["energy_MeV_per_u"])
    return maxnull


def main():
    doms = package_domains()
    maxnull = null_evidence()
    print(f"support map: {len(doms)} channels; null channels: {len(maxnull)}")

    partial = {}
    zeroed_below = 0
    zeroed_zero = 0
    zeroed_above = 0
    escalate = []
    targets_order = None
    for pid in PID_ORDER:
        d = json.load(open(DUMP_RAW / f"{pid}_dump.json"))
        pz, pa = PROJ_ZA[pid]
        assert (d["projectile_z"], d["projectile_a"]) == (pz, pa), pid
        secs = d["sections"]
        assert len(secs) == 25, pid
        grid0 = secs[0]["grid"]
        assert len(grid0) == NE, (pid, len(grid0))
        if targets_order is None:
            targets_order = [el["target_z"] for el in grid0[0]["elements"]]
        assert targets_order == CANON_Z, f"{pid} dump target order != canonical"
        for s, sec in enumerate(secs):
            assert len(sec["grid"]) == NE
            for j, pt in enumerate(sec["grid"]):
                e = EMIN + j * ESTEP
                assert abs(pt["energy_mevu"] - e) < 1e-9, (pid, s, j)
                for el in pt["elements"]:
                    tz = el["target_z"]
                    xs = el["mass_partial_per_mm_at_1g_cm3"]
                    flo, fhi = doms[(pz, pa, tz)]
                    keep = True
                    if xs == 0.0:
                        keep = False
                        zeroed_zero += 1
                    elif e > fhi + 1e-9:
                        keep = False
                        zeroed_above += 1
                    elif e < flo - 1e-9:
                        nul = maxnull.get((pz, pa, tz), -1.0)
                        if e <= nul + 1e-9:
                            keep = False
                            zeroed_below += 1
                        else:
                            escalate.append((pz, pa, tz, e, xs, flo))
                    if not keep:
                        xs = 0.0
                    partial[(pid, s, tz, j)] = xs
    # Masked totals in ascending-target order (bitwise contract with loader).
    total = {}
    for pi, pid in enumerate(PID_ORDER):
        for s in range(25):
            for j in range(NE):
                e = EMIN + j * ESTEP
                tot = 0.0
                for tz in CANON_Z:
                    xs = partial[(pid, s, tz, j)]
                    flo, fhi = doms[PROJ_ZA[pid] + (tz,)]
                    if e < flo or e > fhi:
                        xs = 0.0
                    tot += xs
                total[(pid, s, j)] = tot
    # Gate BEFORE any binary is written.
    need = []
    seen = {}
    for pz, pa, tz, e, xs, flo in sorted(escalate):
        if e - seen.get((pz, pa, tz), -1e9) >= 1.9:
            seen[(pz, pa, tz)] = e
            need.append({"projectile_Z": pz, "projectile_A": pa, "target_Z": tz,
                         "energy_MeV_per_u": round(e, 2), "xs": xs,
                         "channel_floor": flo})
    json.dump({"need_probes": need},
              open(V2 / "step28" / "need_probes.json", "w"), indent=1)
    print(f"NEED probes: {len(need)}; zeroed xs0={zeroed_zero} "
          f"below-null={zeroed_below} above-emax={zeroed_above}")
    if DRY_RUN:
        print("dry-run: binaries NOT written")
        return
    # Residual-NEED policy (campaigns frozen): final compile accepts ONLY the
    # frozen residual list (bitwise match to need_frozen_v2_1.json) via
    # --accept-residual-need. Any NEW need point refuses. Frozen points are
    # kept raw and runtime-masked (partial exactly 0 below floor).
    if need:
        frozen = json.load(open(V2 / "step28" / "need_frozen_v2_1.json"))["need_probes"]
        if need == frozen and "--accept-residual-need" in sys.argv:
            print(f"accepting {len(need)} frozen residual NEED points "
                  f"(runtime-masked, see need_frozen_v2_1.json)")
        else:
            raise RuntimeError(
                f"refusing final compile with {len(need)} unprobed NEED points; "
                f"run probe fills first")
    out_bin = REPO / "data/schneider/secondary_inelastic_rates_v2_1.bin"
    tmpdir = Path(tempfile.mkdtemp(prefix="secratv3_", dir=REPO / "data/schneider"))
    tmp = tmpdir / "secondary_inelastic_rates_v2_1.bin"
    with open(tmp, "wb") as f:
        f.write(b"SCHN2RAT")
        f.write(struct.pack("<I", 3))  # version 3: 14 proj + domain block
        f.write(struct.pack("<IIII", 14, 25, 13, NE))
        f.write(struct.pack("<ddd", EMIN, EMAX, ESTEP))
        for pid in PID_ORDER:
            f.write(struct.pack("<ii", *PROJ_ZA[pid]))
        f.write(struct.pack("<13i", *CANON_Z))
        for pid in PID_ORDER:
            for s in range(25):
                for tz in CANON_Z:
                    for j in range(NE):
                        f.write(struct.pack("<d", partial[(pid, s, tz, j)]))
        for pid in PID_ORDER:
            for s in range(25):
                for j in range(NE):
                    f.write(struct.pack("<d", total[(pid, s, j)]))
        for pid in PID_ORDER:
            pz, pa = PROJ_ZA[pid]
            for tz in CANON_Z:
                flo, fhi = doms[(pz, pa, tz)]
                f.write(struct.pack("<dd", flo, fhi))
                f.write(struct.pack("<B", 1))
                f.write(b"\x00" * 7)
    bin_sha = sha256_file(tmp)
    meta = {
        "schema_version": 1, "format": "SCHN2RAT", "binary_version": 3,
        "data_filename": "secondary_inelastic_rates_v2_1.bin",
        "data_sha256": bin_sha,
        "file_size_bytes": tmp.stat().st_size,
        "binary_magic": "SCHN2RAT",
        "num_projectiles": 14,
        "projectiles": [{"z": PROJ_ZA[pid][0], "a": PROJ_ZA[pid][1]} for pid in PID_ORDER],
        "target_order": CANON_Z,
        "energy_grid": {"emin": EMIN, "emax": EMAX, "step": ESTEP, "count": NE},
        "channel_domains": [
            {"projectile_z": PROJ_ZA[pid][0], "projectile_a": PROJ_ZA[pid][1],
             "target_z": tz,
             "energy_min_mevu": doms[PROJ_ZA[pid] + (tz,)][0],
             "energy_max_mevu": doms[PROJ_ZA[pid] + (tz,)][1],
             "has_support": True}
            for pid in PID_ORDER for tz in CANON_Z],
        "topas_version": "4.2.p3", "geant4_version": "11.03.p02",
        "physics_list": "g4em-standard_opt4 + g4ion-binarycascade",
        "schneider_source_file": "data/HUtoMaterialSchneider.txt",
        "schneider_sha256": sha256_file(REPO / "data/HUtoMaterialSchneider.txt"),
        "zeroing_policy": ("zero iff XS==0 (model threshold), E <= "
                           "per-channel max-null-beam (dynamic Geant4 nulls), "
                           "or E > channel emax (package is the support map); "
                           "substantial-XS-without-nodes NEVER zeroed; "
                           "totals = masked-at-node sums, ascending-target order"),
        "zeroed_xs0": zeroed_zero, "zeroed_below_null": zeroed_below,
        "zeroed_above_emax": zeroed_above,
        "units": "mm, MeV, ns, g/cm^3",
        "generation_timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "compiler_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "compiler_command": "python3 tools/compile_rates_v2_1.py " + " ".join(sys.argv[1:]),
    }
    (tmpdir / "secondary_inelastic_rates_v2_1.metadata.json").write_text(
        json.dumps(meta, indent=1))
    tmp.replace(out_bin)
    (REPO / "data/schneider/secondary_inelastic_rates_v2_1.metadata.json").write_text(
        json.dumps(meta, indent=1))
    import shutil
    shutil.rmtree(tmpdir)
    print(f"compiled {out_bin} ({out_bin.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
