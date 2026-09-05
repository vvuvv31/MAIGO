#!/usr/bin/env python3
"""Attribute slab-escape energy to exit faces (Step06 tail location, offline).

Scope: v2 29-column binary-le ntuples. For every track whose terminal step
carries positive post-step KE, the post position must lie on the slab boundary
with outward direction (same strict rule as the analyzer audit); its KE is
booked to that face, split into C12-primary exit vs electron-family escape.
Re-entry geometries, non-unit weights and non-EM particles are rejected, same
as the analyzer. Escape energy is located, never redistributed.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

COLUMNS = ["run", "event", "track", "parent", "pdg", "step",
           "birth_x_mm", "birth_y_mm", "birth_z_mm", "birth_ke_MeV",
           "pre_x_mm", "pre_y_mm", "pre_z_mm", "post_x_mm", "post_y_mm",
           "post_z_mm", "edep_MeV", "pre_ke_MeV", "post_ke_MeV",
           "weight", "density_g_cm3", "schema_version", "parent_valid",
           "parent_ke_MeV", "parent_dir_x", "parent_dir_y", "parent_dir_z",
           "creator_process_id", "birth_density_g_cm3"]
INT_POS = set(range(6)) | {21, 22, 27}
FACES = ["-x", "+x", "-y", "+y", "-z", "+z"]


def load(steps):
    header = steps.with_suffix(".header").read_text()
    actual = re.findall(r"^\s*([if]\d+):\s*(\S+)\s*$", header, re.MULTILINE)
    expected = [("i4" if i in INT_POS else "f8", n) for i, n in enumerate(COLUMNS)]
    if actual != expected:
        raise ValueError(f"{steps}: not a v2 29-column binary ntuple header")
    dtype = np.dtype([(n, "<i4" if i in INT_POS else "<f8") for i, n in enumerate(COLUMNS)])
    entries = int(re.search(r"Number of Scored Entries: (\d+)", header)[1])
    if steps.stat().st_size != entries * dtype.itemsize:
        raise ValueError(f"{steps}: binary byte count mismatch")
    raw = np.memmap(steps, dtype=dtype, mode="r")
    d = np.column_stack([raw[n] for n in COLUMNS])
    del raw
    if len(d) != entries or not np.isfinite(d).all():
        raise ValueError(f"{steps}: truncated or non-finite ntuple")
    return d


def attribute(d, bounds, tolerance=1e-8):
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    if np.any(d[:, 19] != 1):
        raise ValueError("Escape attribution requires analog unit-weight tracks")
    if np.any(~np.isin(d[:, 4], [1000060120, 11, 22])):
        raise ValueError("Supports EM C12/electron/photon only")
    order = np.lexsort((d[:, 5], d[:, 2], d[:, 1], d[:, 0]))
    ordered = d[order]
    change = np.any(ordered[1:, :3] != ordered[:-1, :3], axis=1)
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:] - 1, len(ordered) - 1]
    faces = {f: {"tracks": 0, "c12_MeV": 0.0, "electron_family_MeV": 0.0,
                 "other_MeV": 0.0} for f in FACES}
    for start, end in zip(starts, ends):
        last = ordered[end]
        if last[18] <= 0:
            continue
        post, pre = last[13:16], last[10:13]
        direction = post - pre
        low = np.abs(post - bounds[:, 0]) <= tolerance
        high = np.abs(post - bounds[:, 1]) <= tolerance
        if not np.all((post >= bounds[:, 0] - tolerance) &
                      (post <= bounds[:, 1] + tolerance)):
            raise ValueError("Positive-KE terminal outside slab: truncated track")
        hit = [i for i in range(3)
               if (low[i] and direction[i] < 0) or (high[i] and direction[i] > 0)]
        if not hit:
            raise ValueError("Terminal boundary direction is not outward")
        face = FACES[2 * hit[0] + (1 if high[hit[0]] else 0)]
        ke = float(last[18])
        faces[face]["tracks"] += 1
        if last[3] == 0 and int(last[4]) == 1000060120:
            faces[face]["c12_MeV"] += ke
        elif int(last[4]) in (11, 22):
            faces[face]["electron_family_MeV"] += ke
        else:
            faces[face]["other_MeV"] += ke
    electron_total = sum(v["electron_family_MeV"] for v in faces.values())
    out = {"faces": faces,
           "electron_escape_total_MeV": electron_total,
           "forward_share": (faces["+z"]["electron_family_MeV"] / electron_total
                             if electron_total > 0 else None)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=Path, required=True)
    p.add_argument("--slab-bounds-mm", type=float, nargs=6, required=True)
    p.add_argument("--tolerance-mm", type=float, default=1e-8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = attribute(load(a.steps), a.slab_bounds_mm, a.tolerance_mm)
    result["scope"] = ("Exit-face attribution of recorded terminal KE; "
                       "locates the tail, redistributes nothing")
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
