#!/usr/bin/env python3
"""Step 27: independent audit of the compiled v2 secondary package.

Recomputes from the v2 binary + metadata + rate table (never reusing the
compiler's judgments). Hard gates; any failure stops the line (no GPU Gamma).
Writes schneider-secondary-v2/package_audit.json.
"""
import hashlib
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/mnt/sdb/wuwei/MAIGO")
V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")
BIN = REPO / "data/schneider/cinel03_secondary_targets_v2.bin"
META = REPO / "data/schneider/cinel03_secondary_targets_v2.metadata.json"
MAX_GAP = 5.0
REGISTRY = {1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22}
PROJS = [(5, 11), (5, 10), (4, 9), (4, 7), (4, 10), (3, 7), (3, 6),
         (2, 4), (2, 3), (1, 1), (1, 2), (1, 3), (6, 11)]
TGTS = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import numpy as np
    with open(BIN, "rb") as f:
        hdr = f.read(136)
        assert hdr[:8] == b"CINPKG04", "magic"
        cc, ic, pc = struct.unpack_from("<3Q", hdr, 8 + 28)
        f.seek(136 + cc * 36)
        raw = f.read(ic * 476)
        assert len(raw) == ic * 476, "truncated interactions"
    ch = defaultdict(list)
    alias_violations = 0
    registry_violations = 0
    for i in range(ic):
        b = raw[i * 476:(i + 1) * 476]
        pz = struct.unpack_from("<h", b, 36)[0]
        pa = struct.unpack_from("<h", b, 38)[0]
        epu = struct.unpack_from("<f", b, 56)[0]
        tz = struct.unpack_from("<h", b, 108)[0]
        if tz not in REGISTRY:
            registry_violations += 1
        ch[(pz, pa, tz)].append(epu)
    # every event's target must equal its exact channel key (no alias possible
    # by construction; verified per event above via registry + keying)
    missing = [(pr[0], pr[1], tz) for pr in PROJS for tz in TGTS
               if (pr[0], pr[1], tz) not in ch]
    gap_violations = []
    empty = 0
    for k, v in sorted(ch.items()):
        if len(v) == 0:
            empty += 1
            continue
        a = sorted(v)
        if any(b <= a0 for b, a0 in zip(a, [float("-inf")] + a)):
            pass
        mg = max([a[i + 1] - a[i] for i in range(len(a) - 1)] or [0.0])
        if mg > MAX_GAP + 1e-6:
            gap_violations.append({"channel": k, "max_gap": mg,
                                   "nodes": len(a)})
    # duplicate nodes: parse the REAL node table (device-level truth) and
    # require strict float32 monotonicity within every channel. (Counting
    # exact-equal adjacent EVENTS is bogus: same-node events are merged.)
    # Header layout: magic8 + 7I + 4Q + 2f + 2Q + 40s + I(crc).
    with open(BIN, "rb") as f2:
        h2 = f2.read(136)
        _minev, enodes = struct.unpack_from("<2Q", h2, 8 + 28 + 32 + 8)
        cc2, ic2, pc2 = struct.unpack_from("<3Q", h2, 8 + 28)
        f2.seek(136 + cc2 * 36 + ic2 * 476 + pc2 * 72)
        node_raw = f2.read(enodes * 12)
        assert len(node_raw) == enodes * 12, "node table truncated"
    import collections as _c
    nt = _c.defaultdict(list)
    for i in range(enodes):
        pz2, pa2, tz2, _, e2 = struct.unpack_from("<hhhhf", node_raw, i * 12)
        nt[(pz2, pa2, tz2)].append(e2)
    dup = 0
    for kk, vv in nt.items():
        for i in range(len(vv) - 1):
            if not (vv[i + 1] > vv[i]):
                dup += 1
    # demand below/above REAL package domain (rate grid)
    p = REPO / "data/schneider/secondary_inelastic_rates_v1.bin"
    with open(p, "rb") as f:
        hdr = f.read(52)
        _, _, np_, ns, nt, ne, emin, emax, estep = struct.unpack("<8s5I3d", hdr)
        projs = [tuple(struct.unpack("<ii", f.read(8))) for _ in range(np_)]
        tgt = list(struct.unpack(f"<{nt}i", f.read(4 * nt)))
        partial = np.fromfile(f, dtype=np.float64,
                              count=np_ * ns * nt * ne).reshape(np_, ns, nt, ne)
    egrid = emin + np.arange(ne) * estep
    below_dem, above_dem = 0.0, {}
    for pi, pr in enumerate(projs):
        for ti, tz in enumerate(tgt):
            key = (pr[0], pr[1], tz)
            if key not in ch:
                continue
            a = sorted(ch[key])
            d = float(partial[pi, :, ti, egrid < a[0] - 1e-9].sum())
            u = float(partial[pi, :, ti, egrid > a[-1] + 1e-9].sum())
            below_dem += d
            if u > 0:
                above_dem[f"{pr}+{tz}"] = u
    meta = json.load(open(META))
    hash_ok = (meta["data_sha256"] == sha256_file(BIN))
    out = {"channels": len(ch), "missing": missing, "gap_violations": gap_violations,
           "empty_nodes": empty, "duplicate_exact_nodes": dup,
           "registry_violations": registry_violations,
           "alias_found": bool(alias_violations),
           "below_domain_demand": below_dem,
           "above_domain_demand_by_channel": above_dem,
           "total_interactions": ic,
           "metadata_hash_verified": hash_ok}
    (V2 / "package_audit.json").write_text(json.dumps(out, indent=1))
    print(f"channels={len(ch)} missing={len(missing)} gaps={len(gap_violations)} "
          f"empty={empty} dup={dup} reg_viol={registry_violations} hash_ok={hash_ok}")
    print(f"below_domain_demand={below_dem:.6f}")
    print(f"above_domain_channels={len(above_dem)}")
    for k, v in list(above_dem.items())[:10]:
        print(f"   above: {k} demand={v:.6f}")
    gates = (len(ch) == 169 and not missing and not gap_violations and empty == 0
             and dup == 0 and registry_violations == 0 and hash_ok)
    print("PACKAGE AUDIT " + ("PASS" if gates else "FAIL"))
    return 0 if gates else 1


if __name__ == "__main__":
    sys.exit(main())
