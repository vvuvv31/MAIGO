#!/usr/bin/env python3
"""Adaptive-lattice gamma re-evaluation on frozen fullplan dose pairs.

The frozen evaluator (evaluate_topas10x_gamma: 0.5 mm search lattice) is kept
unchanged for provenance. This tool reuses its pass_mask verbatim for the
coarse stage, then re-searches ONLY the coarse failures on a finer lattice
(default 0.25 mm) with the same trilinear interpolant. Refinement can only
rescue coarse failures, never overturn coarse passes.

Rationale: the gamma index is defined as the minimum over continuous space
(Low et al.). A 0.5 mm lattice places only ~33 points inside a 1 mm ball, so
in high-gradient regions (penumbra, distal falloff, interfaces) the coarse
lattice systematically misses the minimizing location and underestimates the
true pass rate. Denser sampling of the same interpolant is a strictly more
accurate evaluation of the same metric, not a looser one: every lattice
point is a feasible point of the continuous field, so the refined pass rate
remains a lower bound on the continuous-minimum pass rate.

Writes <folder>/gamma_refined.json; refuses to overwrite. Never modifies
frozen gamma.json / dose inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from evaluate_topas10x_gpu_gamma import pass_mask
from run_topas10x_gpu_benchmark import sha, save


def refine_failures(evaluation, reference, pts, spacing, dose_percent, dta,
                    local, coarse_passed, step=0.25):
    """Re-search coarse failures on a finer lattice. Returns updated passes."""
    rr = reference[tuple(pts.T)]
    tol = dose_percent * 0.01 * (rr if local else float(reference.max()))
    passed = coarse_passed.copy()
    k = int(round(dta / step))
    offsets = []
    for z in range(-k, k + 1):
        for y in range(-k, k + 1):
            for x in range(-k, k + 1):
                dist = step * np.sqrt(x * x + y * y + z * z)
                if 0 < dist <= dta + 1e-12:
                    offsets.append((dist, np.array([z, y, x]) * step / spacing))
    indices = np.flatnonzero(~passed)
    for dist, offset in sorted(offsets, key=lambda v: v[0]):
        if not len(indices):
            break
        coords = pts[indices].T.astype(float) + offset[:, None]
        valid = np.all((coords >= 0) & (coords <= np.array(evaluation.shape)[:, None] - 1), axis=0)
        values = map_coordinates(evaluation, coords, order=1, mode="constant",
                                 cval=np.nan, prefilter=False)
        denom = tol[indices] if local else tol
        hit = valid & ((dist / dta) ** 2 + np.square((values - rr[indices]) / denom) <= 1)
        passed[indices[hit]] = True
        indices = indices[~hit]
    return passed


def evaluate_refined(folder, step=0.25):
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "execution.json").read_text())
    frozen = json.loads((folder / "gamma.json").read_text())
    if s["status"] != "complete":
        raise ValueError("Incomplete case")
    if sha(folder / "gpu_sum.raw") != s["aggregate_sha256"]:
        raise ValueError("GPU sum SHA mismatch")
    if sha(folder / "topas_sum.raw") != m["reference_sum_sha256"]:
        raise ValueError("Reference SHA mismatch")
    if sum(x["histories"] for x in s["completed"]) != m["histories"]:
        raise ValueError("Incomplete histories")
    g = np.fromfile(folder / "gpu_sum.raw", dtype="<f4").reshape(m["gpu_shape_zyx"])
    if m["mapping"] == "packed_xneg":
        g = np.flip(g.transpose(1, 2, 0), axis=2)
    elif m["mapping"] != "native":
        raise ValueError("Unknown geometry mapping")
    r = np.fromfile(folder / "topas_sum.raw", dtype="<f4").reshape(m["topas_shape_zyx"])
    if r.shape != g.shape:
        raise ValueError("Shape mismatch")
    mask = r >= 0.1 * r.max()
    pts = np.argwhere(mask)
    if not len(pts):
        raise ValueError("Empty mask")
    sample = np.random.default_rng(42).choice(len(pts), min(50000, len(pts)), replace=False)
    results = {}
    for dd, dta in [(3, 3), (2, 2), (1, 1), (3, 0)]:
        for local in [False, True]:
            key = f"{'local' if local else 'global'}_{dd}pct_{dta}mm"
            coarse = pass_mask(g, r, pts, np.array(m["spacing_zyx"]), dd, dta, local)
            expected = frozen["gamma"][key]["full_mask_percent"]
            if abs(100 * float(coarse.mean()) - expected) > 1e-9:
                raise ValueError(f"Frozen reproduction failed for {key}")
            if dta == 0 or step >= 0.5:
                refined = coarse
                rescued = 0
            else:
                refined = refine_failures(g, r, pts, np.array(m["spacing_zyx"]),
                                          dd, dta, local, coarse, step=step)
                rescued = int(refined.sum() - coarse.sum())
            results[key] = {"full_mask_percent": 100 * float(refined.mean()),
                            "passed": int(refined.sum()), "evaluated": len(pts),
                            "sample50k_percent": 100 * float(refined[sample].mean()),
                            "coarse_full_mask_percent": 100 * float(coarse.mean()),
                            "rescued_by_refinement": rescued}
            print(m["case"], key, results[key], flush=True)
    out = {"case": m["case"], "histories_gpu": m["histories"],
           "histories_topas": frozen["histories_topas"], "dose_scale": 1.0,
           "reference": frozen["reference"],
           "method": {**frozen["method"],
                      "refinement": f"coarse failures re-searched on {step} mm lattice, same trilinear interpolant",
                      "coarse_step_mm": 0.5, "refine_step_mm": step,
                      "guarantee": "refinement only rescues coarse failures; refined rate lower-bounds continuous-minimum rate"},
           "gamma": results,
           "dose_sum_ratio": frozen["dose_sum_ratio"],
           "in_mask_dose_sum_ratio": frozen["in_mask_dose_sum_ratio"],
           "high_dose_correlation": frozen["high_dose_correlation"],
           "physics_scope": frozen["physics_scope"],
           "queue_overflow_accepted_shards": 0,
           "overflow_attempts_excluded": s["overflow_attempts"],
           "frozen_gamma_sha256": sha(folder / "gamma.json"),
           "source_sha256": sha(__file__),
           "gpu_sha256": sha(folder / "gpu_sum.raw"),
           "reference_sha256": sha(folder / "topas_sum.raw")}
    target = folder / "gamma_refined.json"
    if target.exists():
        raise ValueError("Refuse overwrite gamma_refined")
    save(target, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--step", type=float, default=0.25)
    a = p.parse_args()
    evaluate_refined(a.folder, step=a.step)
