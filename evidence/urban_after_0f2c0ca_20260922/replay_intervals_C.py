#!/usr/bin/env python3
"""Urban-B (post B1-B6) water-replay interval analysis.

1. B-candidate moments vs TOPAS job-6529 absolute reference (same metric
   definitions as /tmp/paired_replay_intervals.py and
   /tmp/topas_replay_intervals.py).
2. B vs OLD-Urban paired on source_history (same entrance, same seed):
   isolates the B1-B6 effect size per history.
"""
import numpy as np

B = "/mnt/sda/wuwei/urban_C2_validation/planes/replay_primary_planes_C.csv"
OLD = "/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/gpu_urban_v2/primary_planes.csv"
DEPTHS = [40.0, 60.0, 80.0, 100.0, 120.0]

# TOPAS absolute anchors (job 6529 parent-0, urban.md §paired replay).
TOPAS = {
    0: dict(angVar=37.89, disVar=0.00404, q999=32.29),
    1: dict(angVar=47.70, disVar=0.00505, q999=33.70),
    2: dict(angVar=69.35, disVar=0.00711, q999=36.62),
    3: dict(angVar=169.51, disVar=0.01378, q999=49.76),
}


def load(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1,
                   usecols=(0, 1, 2, 4, 5, 6, 7, 8))
    order = np.lexsort((d[:, 1], d[:, 0]))
    return d[order]


def paired_intervals(d):
    out = []
    for k in range(4):
        a = d[d[:, 1] == k]
        b = d[d[:, 1] == k + 1]
        bh = set(b[:, 0].astype(np.int64).tolist())
        m = np.array([h in bh for h in a[:, 0]])
        a = a[m]
        b = b[np.isin(b[:, 0], a[:, 0])]
        oa = np.argsort(a[:, 0])
        ob = np.argsort(b[:, 0])
        a, b = a[oa], b[ob]
        assert np.array_equal(a[:, 0], b[:, 0])
        out.append((a, b))
    return out


def moments(a, b):
    dz = b[:, 2] - a[:, 2]
    dtx = b[:, 5] - a[:, 5]
    dty = b[:, 6] - a[:, 6]
    dth2 = dtx ** 2 + dty ** 2
    rx = (b[:, 3] - a[:, 3]) - a[:, 5] * dz
    ry = (b[:, 4] - a[:, 4]) - a[:, 6] * dz
    dr2 = rx ** 2 + ry ** 2
    th = np.sqrt(dth2)
    q = np.quantile(th, [0.68, 0.99, 0.999]) * 1000.0
    return (np.mean(dth2) * 1e6, np.mean(dr2), q[0], q[1], q[2], len(a),
            dth2, dr2)


iv_b = paired_intervals(load(B))
iv_old = paired_intervals(load(OLD))
print("interval: B angVar/disVar/q68/q99/q999 + B/TOPAS ratios")
for k in range(4):
    av, dv, q68, q99, q999, n, _, _ = moments(*iv_b[k])
    t = TOPAS[k]
    print(f"{DEPTHS[k]:.0f}->{DEPTHS[k+1]:.0f} N={n} angVar={av:.4f} "
          f"({av/t['angVar']:.3f}) disVar={dv:.6f} ({dv/t['disVar']:.3f}) "
          f"q68={q68:.3f} q99={q99:.3f} q999={q999:.3f} ({q999/t['q999']:.3f})",
          flush=True)
print("paired B vs OLD-Urban (same histories): mean/var ratios + "
      "per-history |dth| correlation", flush=True)
for k in range(4):
    ab, bb = iv_b[k], iv_old[k]
    # common histories
    ha = set(ab[0][:, 0].astype(np.int64).tolist())
    hb = set(bb[0][:, 0].astype(np.int64).tolist())
    common = ha & hb
    ma = np.isin(ab[0][:, 0], list(common))
    mb = np.isin(bb[0][:, 0], list(common))
    A1, B1 = ab[0][ma], ab[1][ma]
    A2, B2 = bb[0][mb], bb[1][mb]
    o1, o2 = np.argsort(A1[:, 0]), np.argsort(A2[:, 0])
    A1, B1, A2, B2 = A1[o1], B1[o1], A2[o2], B2[o2]
    assert np.array_equal(A1[:, 0], A2[:, 0])
    _, _, _, _, _, _, dth2_b, dr2_b = moments(A1, B1)
    _, _, _, _, _, _, dth2_o, dr2_o = moments(A2, B2)
    th_b = np.sqrt(dth2_b)
    th_o = np.sqrt(dth2_o)
    corr = np.corrcoef(th_b, th_o)[0, 1]
    print(f"{DEPTHS[k]:.0f}->{DEPTHS[k+1]:.0f} Ncommon={len(A1)} "
          f"angVar B/OLD={np.mean(dth2_b)/np.mean(dth2_o):.5f} "
          f"disVar B/OLD={np.mean(dr2_b)/np.mean(dr2_o):.5f} "
          f"corr(|dth|)={corr:.5f}", flush=True)
