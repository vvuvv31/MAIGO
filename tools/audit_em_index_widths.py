#!/usr/bin/env python3
"""Structural audit of the unified-EM exact-search buckets (candidate A gating).

Reads the validated EMJOINT1 package and, for the current 8-bit exponent index
and a proposed exponent+2-bit-mantissa index, reports the candidate interval
width each bucket leaves for the binary search. This is a table-structure
audit, not a runtime measurement.

Usage: python3 tools/audit_em_index_widths.py data/em/unified_em_v1.bin
"""
import argparse
import struct
from pathlib import Path

import numpy as np

NODE_STRUCT = np.dtype([("energy", "<f4"), ("rest", "<f4", 12)])
SEG_STRUCT = np.dtype([("lower", "<f4"), ("rest", "<f4", 5)])
REC_STRUCT = np.dtype([
    ("mass", "<f4"), ("ratio", "<f4"), ("cut", "<f4"), ("excitation", "<f4"),
    ("e0", "<f4"), ("spin", "<f4"), ("step_fraction", "<f4"),
    ("final_range", "<f4"), ("linear_limit", "<f4"), ("is_ion", "<i4"),
    ("form_factor", "<f4"), ("magnetic_moment2", "<f4"), ("lowest_kinetic", "<f4"),
    ("peak", "<f4"), ("z", "<i4"), ("a", "<i4"), ("node_count", "<u4"),
    ("node_offset", "<u4"), ("counts", "<u4", 4), ("offsets", "<u4", 4),
    ("_pad", "<u1", 8),
])


def edges_8bit():
    e = np.empty(257, dtype=np.float64)
    e[0] = 0.0
    e[1:256] = np.ldexp(1.0, np.arange(1, 256) - 127)
    e[255] = np.ldexp(1.0, 128)  # bucket 255 edge marker, replaced by +inf
    e[256] = np.inf
    return e


def edges_10bit():
    exp = np.arange(0, 256)
    man = np.arange(4) / 4.0
    base = np.ldexp(1.0, exp - 127)
    e = (base[:, None] * (1.0 + man[None, :])).reshape(-1).astype(np.float64)
    return np.concatenate([[0.0], e, [np.inf]])


def widths(lower, edges):
    # index[b] = first position with lower >= edges[b]; mirrors build_unified_em_index
    idx = np.searchsorted(lower, edges, side="left").astype(np.int64)
    lo = np.maximum(idx[:-1] - 1, 0)
    hi = idx[1:]
    w = np.maximum(hi - lo, 1)
    return w


def summarize(name, all_w):
    w = np.concatenate(all_w) if all_w else np.array([1])
    comp = np.ceil(np.log2(np.maximum(w, 1) + 1))
    print(f"{name}: buckets={w.size} mean_width={w.mean():.3f} "
          f"median={np.median(w):.0f} p90={np.percentile(w,90):.0f} "
          f"max={w.max()} frac_width_gt2={np.mean(w>2)*100:.1f}% "
          f"mean_bsearch_steps={comp.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    a = ap.parse_args()
    path = Path(a.package)
    with path.open("rb") as f:
        if f.read(8) != b"EMJOINT1":
            raise SystemExit("not an EMJOINT1 package")
        version, mats, species, recs, nodes, segs = struct.unpack("<6I", f.read(24))
    print(f"materials={mats} records={recs} nodes={nodes} segments={segs}")
    base = 32 + 8 * mats + 8 * species
    rec_arr = np.memmap(path, dtype=REC_STRUCT, mode="r", offset=base, shape=(recs,))
    node_arr = np.memmap(path, dtype=NODE_STRUCT, mode="r",
                         offset=base + 104 * recs, shape=(nodes,))
    seg_arr = np.memmap(path, dtype=SEG_STRUCT, mode="r",
                        offset=base + 104 * recs + 52 * nodes, shape=(segs,))
    node_e = node_arr["energy"]
    seg_l = seg_arr["lower"]
    e8 = edges_8bit()
    e10 = edges_10bit()
    w8, w10 = [], []
    for r in range(recs):
        off, cnt = int(rec_arr["node_offset"][r]), int(rec_arr["node_count"][r])
        lower = np.asarray(node_e[off:off + cnt], dtype=np.float64)
        if lower.size >= 2:
            w8.append(widths(lower, e8))
            w10.append(widths(lower, e10))
        for k in range(4):
            c = int(rec_arr["counts"][r][k])
            so = int(rec_arr["offsets"][r][k])
            if c >= 2:
                sl = np.asarray(seg_l[so:so + c], dtype=np.float64)
                w8.append(widths(sl, e8))
                w10.append(widths(sl, e10))
    summarize("8-bit exponent index", w8)
    summarize("10-bit exponent+2mantissa", w10)


if __name__ == "__main__":
    main()
