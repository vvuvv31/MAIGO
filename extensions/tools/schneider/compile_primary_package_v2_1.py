#!/usr/bin/env python3
"""Primary CINEL03 v2.1 compiler: v1 C12 production raw (stride-preserved)
plus validated new low-E C12 fills (stride-1, every event kept).

Rules (mirror Step-17 audit, no silent changes):
- v1 files: same sorted glob + stride=20 as the v1 compile; the v1 subset
  MUST reproduce exactly 30113 interactions (asserted).
- new fills: validated tasks only (raw_validation_all.json), stride 1.
- closure bound e_total <= e_coll + max(200, 0.2*e_coll); C12 projectiles only.
- sort (pz,pa,tz,E); 1-MeV cells with bin-0 clamp for sub-0.5 events (same
  rule as the secondary v2 compiler); exact-float32 energy nodes.
- temp path + re-read + atomic rename.
Outputs (new names; v1 untouched):
  data/schneider/cinel03_c12_targets_v2_1.bin (+ .metadata.json, +.channels.json)
"""
import datetime
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3]))
WORK = Path(os.environ.get("MAIGO_PACKAGE_WORK_ROOT", REPO / "extensions" / "work"))
V2 = Path(os.environ.get("MAIGO_V2_CAMPAIGN_ROOT", WORK / "schneider-secondary-v2"))
V1_PROD = Path(os.environ.get("MAIGO_PRIMARY_CAMPAIGN_ROOT", WORK / "production"))
V1_COUNT = 30113
V1_UUID = "00000000-0000-4000-8000-000000000017"
V2_1_UUID = "00000000-0000-4000-8000-000000000025"
STRIDE_V1 = 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main():
    sys.path.insert(0, str(REPO / "extensions" / "package_tools"))
    import cinel02
    import cinel03
    v1_files = sorted(V1_PROD.glob("**/worker_*.cinel02"))
    print(f"v1 C12 raw files: {len(v1_files)}")
    events = []
    for wf in v1_files:
        for rec, prods in cinel02.read_raw(wf):
            if rec["projectile_z"] != 6 or rec["projectile_a"] != 12:
                raise ValueError(f"Non-carbon projectile in {wf}")
            e_coll = float(rec["collision_energy_MeV"])
            e_tot = (float(rec["parent_energy_MeV"]) +
                     float(rec["process_local_deposit_MeV"]) +
                     float(rec["unsupported_product_energy_MeV"]) +
                     sum(float(p["kinetic_energy_MeV"]) for p in prods))
            if e_tot > e_coll + max(200.0, 0.20 * e_coll):
                continue
            events.append((rec, prods))
    v1_pool = len(events)
    events = events[::STRIDE_V1]
    print(f"v1 pool {v1_pool} -> strided {len(events)} (package has {V1_COUNT})")
    assert len(events) == V1_COUNT, \
        f"v1 subset reproduction failed: {len(events)} != {V1_COUNT}"

    # new validated C12 fills (stride 1)
    val = json.load(open(V2 / "raw_validation_all.json"))
    man = json.load(open(V2 / "campaign_manifest_all.json"))
    by_task = {t["task_index"]: t for t in man["tasks"]}
    n_new = 0
    for r in val["results"]:
        if not r["pass"]:
            continue
        t = by_task[r["task_index"]]
        if (t["projectile_Z"], t["projectile_A"]) != (6, 12):
            continue
        for fp in sorted((V2 / t["case_dir"]).rglob("worker_*.cinel02")):
            for rec, prods in cinel02.read_raw(fp):
                e_coll = float(rec["collision_energy_MeV"])
                e_tot = (float(rec["parent_energy_MeV"]) +
                         float(rec["process_local_deposit_MeV"]) +
                         float(rec["unsupported_product_energy_MeV"]) +
                         sum(float(p["kinetic_energy_MeV"]) for p in prods))
                if e_tot > e_coll + max(200.0, 0.20 * e_coll):
                    continue
                events.append((rec, prods))
                n_new += 1
    print(f"new C12 events: {n_new}; total: {len(events)}")
    if n_new == 0:
        raise RuntimeError("no validated new C12 events; refusing empty-delta compile")

    events.sort(key=lambda it: (it[0]["projectile_z"], it[0]["projectile_a"],
                                it[0]["target_z"],
                                f32(it[0]["collision_energy_MeV_per_u"])))
    packed_interactions, packed_products, cells = [], [], []

    def cell_bin(e):
        return max(0, int(math.floor((e - 0.5) / 1.0)))

    cursor = 0
    while cursor < len(events):
        r0 = events[cursor][0]
        key = (r0["projectile_z"], r0["projectile_a"], r0["target_z"],
               cell_bin(r0["collision_energy_MeV_per_u"]))
        start = cursor
        while cursor < len(events):
            r = events[cursor][0]
            k = (r["projectile_z"], r["projectile_a"], r["target_z"],
                 cell_bin(r["collision_energy_MeV_per_u"]))
            if k != key:
                break
            cursor += 1
        cells.append(key + (start, cursor - start))
    for (pz, pa, tz, ebin, start, count) in cells:
        for i in range(start, start + count):
            rec, prods = events[i]
            packed_interactions.append(cinel02._pack_fixed(rec))
            for p in prods:
                packed_products.append(cinel02._pack_product(p))

    pkg = cinel03.Cinel03Package()
    pkg.minimum_energy_MeV_per_u = 0.5
    pkg.energy_bin_width_MeV_per_u = 1.0
    pkg.minimum_events_per_bin = 1
    pkg.campaign_uuid = V2_1_UUID
    pkg.cells = [{"projectile_z": c[0], "projectile_a": c[1],
                  "target_element_z": c[2], "energy_bin": c[3],
                  "interaction_offset": c[4], "interaction_count": c[5],
                  "energy_lower_MeV_per_u": 0.5 + c[3] * 1.0,
                  "energy_upper_MeV_per_u": 0.5 + (c[3] + 1) * 1.0} for c in cells]
    pkg.interactions = packed_interactions
    pkg.products = packed_products
    cursor = 0
    while cursor < len(events):
        r0 = events[cursor][0]
        nk = (r0["projectile_z"], r0["projectile_a"], r0["target_z"],
              f32(r0["collision_energy_MeV_per_u"]))
        start = cursor
        while cursor < len(events):
            r = events[cursor][0]
            if (r["projectile_z"], r["projectile_a"], r["target_z"],
                    f32(r["collision_energy_MeV_per_u"])) != nk:
                break
            cursor += 1
        pkg.energy_nodes.append(nk)
        pkg.event_offsets.append(start)
    pkg.event_offsets.append(len(events))
    pkg.event_indices = list(range(len(events)))

    tmpdir = Path(tempfile.mkdtemp(prefix="cinel03c12v21_", dir=REPO / "data/schneider"))
    tmp_bin = tmpdir / "cinel03_c12_targets_v2_1.bin"
    pkg.write_binary(tmp_bin)
    reread = cinel03.Cinel03Package.read_binary(tmp_bin)
    assert len(reread.interactions) == len(events), "re-read mismatch"
    final_bin = REPO / "data/schneider/cinel03_c12_targets_v2_1.bin"
    tmp_bin.replace(final_bin)
    import shutil
    shutil.rmtree(tmpdir)
    from collections import defaultdict as dd
    ch = dd(list)
    for n in pkg.energy_nodes:
        ch[(n[0], n[1], n[2])].append(n[3])
    channels = []
    for k in sorted(ch.items()):
        a = sorted(k[1])
        gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        channels.append({
            "projectile_z": k[0][0], "projectile_a": k[0][1],
            "target_element_z": k[0][2],
            "energy_min_MeV_per_u": a[0], "energy_max_MeV_per_u": a[-1],
            "energy_nodes": len(a),
            "maximum_observed_node_gap_MeV_per_u": max(gaps) if gaps else 0.0,
            "maximum_allowed_node_gap_MeV_per_u": 5.0,
            "interpolation_policy": "stochastic_bracketing",
            "target_alias_allowed": False})
    (REPO / "data/schneider/cinel03_c12_targets_v2_1.channels.json").write_text(
        json.dumps({"channels": channels}, indent=1))
    meta = {"schema_version": 1, "format": "CINPKG04",
            "data_filename": "cinel03_c12_targets_v2_1.bin",
            "data_sha256": sha256_file(final_bin),
            "file_size_bytes": final_bin.stat().st_size,
            "topas_version": "4.2.p3", "geant4_version": "geant4-11-03-patch-02",
            "physics_list": "FTFP_INCLXX",
            "schneider_source_path": "data/HUtoMaterialSchneider.txt",
            "schneider_sha256": sha256_file(REPO / "data/HUtoMaterialSchneider.txt"),
            "v1_base_interactions": V1_COUNT,
            "new_events": n_new,
            "total_interactions": len(events),
            "campaign_uuid": V2_1_UUID,
            "compiler_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
            "compiler_command": "python3 tools/compile_primary_package_v2_1.py",
            "units": "mm, MeV, ns",
            "generation_timestamp_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat()}
    (REPO / "data/schneider/cinel03_c12_targets_v2_1.metadata.json").write_text(
        json.dumps(meta, indent=1))
    print(f"compiled {final_bin} ({final_bin.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
