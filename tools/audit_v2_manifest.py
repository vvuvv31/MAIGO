#!/usr/bin/env python3
"""Step 22 independent audit of campaign_manifest.json.

Recomputes, from manifest + existing packages + rate table (never reusing
the generator's internal lists):
  missing_channels == 0            (all 169 (proj,target) planned)
  max_gap_planned <= 5.0           (existing nodes UNION planned new nodes)
  planned_below_domain_demand == 0 (rate-grid demand below planned floor)
  planned_above_domain_demand == 0 (rate-grid demand above planned ceil)
Plus C12 floor coverage. Tier-L probe tasks are measurement-only and excluded
from package-coverage gates (reported separately).
Informational (non-gating): sub-0.5 device-clamp query risk -> Tier-L + Step 28b.
"""
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/mnt/sdb/wuwei/MAIGO")
V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")
MAX_GAP = 5.0


def load_nodes(bin_path):
    with open(bin_path, "rb") as f:
        h = f.read(136)
        cc, ic, pc = struct.unpack_from("<3Q", h, 8 + 28)
        f.seek(136 + cc * 36)
        raw = f.read(ic * 476)
    chans = defaultdict(list)
    for i in range(ic):
        b = raw[i * 476:(i + 1) * 476]
        pz = struct.unpack_from("<h", b, 36)[0]
        pa = struct.unpack_from("<h", b, 38)[0]
        epu = struct.unpack_from("<f", b, 56)[0]
        tz = struct.unpack_from("<h", b, 108)[0]
        chans[(pz, pa, tz)].append(epu)
    return {k: sorted(v) for k, v in chans.items()}


def load_rates():
    import numpy as np
    p = REPO / "data/schneider/secondary_inelastic_rates_v1.bin"
    with open(p, "rb") as f:
        hdr = f.read(52)
        magic, ver, np_, ns, nt, ne, emin, emax, estep = struct.unpack("<8s5I3d", hdr)
        projs = [tuple(struct.unpack("<ii", f.read(8))) for _ in range(np_)]
        tgt = list(struct.unpack(f"<{nt}i", f.read(4 * nt)))
        partial = np.fromfile(f, dtype=np.float64,
                              count=np_ * ns * nt * ne).reshape(np_, ns, nt, ne)
    dem = {}
    for pi, pr in enumerate(projs):
        for ti, tz in enumerate(tgt):
            dem[(pr[0], pr[1], tz)] = partial[pi, :, ti, :]
    return dem, emin, estep, ne


def main():
    import numpy as np
    manifest = json.load(open(V2 / "campaign_manifest.json"))
    sec_nodes = load_nodes(REPO / "data/schneider/cinel03_secondary_targets.bin")
    c12_nodes = load_nodes(REPO / "data/schneider/cinel03_c12_targets.bin")
    demand, emin, estep, ne = load_rates()
    egrid = emin + np.arange(ne) * estep

    planned = defaultdict(list)
    for t in manifest["tasks"]:
        if t["kind"] == "tierL-subhalf-probe":
            continue
        planned[(t["projectile_Z"], t["projectile_A"], t["target_Z"])].append(
            t["energy_MeV_per_u"])

    PROJS = [(5, 11), (5, 10), (4, 9), (4, 7), (4, 10), (3, 7), (3, 6),
             (2, 4), (2, 3), (1, 1), (1, 2), (1, 3), (6, 11)]
    TGTS = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22]
    missing, gap_fail, below_dem, above_dem = 0, [], 0.0, 0.0
    for pr in PROJS:
        for tz in TGTS:
            key = (pr[0], pr[1], tz)
            nodes = sorted(sec_nodes.get(key, []) + planned.get(key, []))
            if not nodes:
                missing += 1
                continue
            gaps = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
            mg = max(gaps) if gaps else 0.0
            if mg > MAX_GAP + 1e-6:
                gap_fail.append((key, mg))
            d = demand.get(key)
            if d is not None:
                below_dem += float(d[:, egrid < nodes[0] - 1e-9].sum())
                above_dem += float(d[:, egrid > nodes[-1] + 1e-9].sum())
    # C12 floors
    c12_fail = []
    for key, nodes in c12_nodes.items():
        new_e = [t["energy_MeV_per_u"] for t in manifest["tasks"]
                 if (t["projectile_Z"], t["projectile_A"], t["target_Z"]) == key]
        lo = min(nodes + new_e)
        if lo > 0.5 + 1e-6:
            c12_fail.append((key, lo))

    ok = (missing == 0 and not gap_fail and below_dem == 0.0 and above_dem == 0.0
          and not c12_fail)
    print(f"missing_channels = {missing} (gate 0)")
    print(f"max_gap violations = {len(gap_fail)} (gate 0) {gap_fail[:3]}")
    print(f"planned_below_domain_demand = {below_dem} (gate 0)")
    print(f"planned_above_domain_demand = {above_dem} (gate 0)")
    print(f"c12 floor violations = {len(c12_fail)} (gate 0)")
    print("AUDIT " + ("PASS" if ok else "FAIL"))
    (V2 / "manifest_audit.json").write_text(json.dumps({
        "missing_channels": missing, "gap_violations": gap_fail,
        "planned_below_domain_demand": below_dem,
        "planned_above_domain_demand": above_dem,
        "c12_floor_violations": c12_fail, "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
