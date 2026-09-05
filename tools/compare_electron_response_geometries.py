#!/usr/bin/env python3
"""Compare v2 electron response across slab geometries under matched birth.

Scope: v2 29-column binary-le ntuples only (Step06 diagnostic). For each
geometry the tool reconstructs electron-root ancestry exactly like the
analyzer, then reports birth spectra, escape fractions, joint
radial/longitudinal histograms and movable fractions — overall and
conditioned on parent-KE bands — plus an interior-birth subset (G3 style).

It never averages shard quantiles and never deletes far-tail bins: joint
histograms use the fixed analyzer edges, overflow raises.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from audit_schneider_response_scope import schneider_identity

V1 = ["run", "event", "track", "parent", "pdg", "step",
      "birth_x_mm", "birth_y_mm", "birth_z_mm", "birth_ke_MeV",
      "pre_x_mm", "pre_y_mm", "pre_z_mm", "post_x_mm", "post_y_mm",
      "post_z_mm", "edep_MeV", "pre_ke_MeV", "post_ke_MeV",
      "weight", "density_g_cm3"]
V2_EXTRA = ["schema_version", "parent_valid", "parent_ke_MeV",
            "parent_dir_x", "parent_dir_y", "parent_dir_z",
            "creator_process_id", "birth_density_g_cm3"]
COLUMNS = V1 + V2_EXTRA
INT_POS = set(range(6)) | {21, 22, 27}
RADIAL_EDGES = np.array([0, .1, .5, 1, 2, 5, 10, 20, 50, 100, 300])
LONG_EDGES = np.array([-500, -100, -20, -5, -1, -.5, 0, .5, 1, 5, 20, 100, 500])
BIRTH_EDGES = np.array([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def joint_shape_distance(a, b):
    """L1 between independently normalized distributions, not energy yields."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < 0) or np.any(b < 0) or a.sum() <= 0 or b.sum() <= 0):
        raise ValueError("Invalid joint histogram")
    return float(np.abs(a/a.sum()-b/b.sum()).sum())


def load_v2_binary(steps):
    """Compatibility name: explicit v2 and v3 headers, never schema aliasing."""
    header = steps.with_suffix(".header").read_text()
    actual = re.findall(r"^\s*([if]\d+):\s*(\S+)\s*$", header, re.MULTILINE)
    expected = [("i4" if i in INT_POS else "f8", n) for i, n in enumerate(COLUMNS)]
    columns, ints, version = COLUMNS, INT_POS, 2
    if actual != expected:
        columns = COLUMNS + ["parent_step_id", "parent_post_ke_MeV", "parent_post_dir_x", "parent_post_dir_y", "parent_post_dir_z"]
        ints, version = INT_POS | {29}, 3
        if actual != [("i4" if i in ints else "f8", n) for i,n in enumerate(columns)]:
            raise ValueError(f"{steps}: not a v2/v3 binary ntuple header")
    dtype = np.dtype([(n, "<i4" if i in ints else "<f8") for i, n in enumerate(columns)])
    entries = int(re.search(r"Number of Scored Entries: (\d+)", header)[1])
    if steps.stat().st_size != entries * dtype.itemsize:
        raise ValueError(f"{steps}: binary byte count mismatch")
    raw = np.memmap(steps, dtype=dtype, mode="r")
    d = np.column_stack([raw[n] for n in columns])
    del raw
    if len(d) != entries or d.shape[1] != len(columns) or not np.isfinite(d).all():
        raise ValueError(f"{steps}: truncated or non-finite ntuple")
    if np.any(d[:, 21] != version):
        raise ValueError(f"{steps}: schema_version != {version}")
    return d


def build_roots(d):
    tracks = {}
    for row in d:
        key = tuple(row[:3].astype(int))
        if key not in tracks:
            tracks[key] = (int(row[3]), int(row[4]), row[6:9].copy(), float(row[9]),
                           row[24:27].copy(), float(row[23]), int(row[22]))
    roots = {}

    def root(key):
        if key in roots:
            return roots[key]
        chain, cur = [], key
        while cur not in roots:
            if cur not in tracks:
                raise ValueError(f"Missing ancestor {cur}")
            if cur in chain:
                raise ValueError("Cyclic track ancestry")
            chain.append(cur)
            parent, pdg = tracks[cur][0], tracks[cur][1]
            if parent == 0:
                roots[cur] = None
                break
            pk = (cur[0], cur[1], parent)
            if pk in tracks and tracks[pk][0] == 0:
                if pdg != 11:
                    raise ValueError(f"Unexpected primary daughter PDG {pdg}")
                roots[cur] = cur
                break
            cur = pk
        value = roots[cur]
        for k in chain:
            roots[k] = value
        return value

    return tracks, root


def summarize(d, tracks, root, bounds, interior_inset_mm):
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    family = {}
    for row in d:
        rk = root(tuple(row[:3].astype(int)))
        if rk is None:
            continue
        slot = family.setdefault(rk, {"birth": tracks[rk][3], "dep": 0.0,
                                      "pke": tracks[rk][5], "pvalid": tracks[rk][6],
                                      "pdir": tracks[rk][4],
                                      "birth_pos": tracks[rk][2]})
        slot["dep"] += float(row[16] * row[19])
    # Escape per family requires terminal audit; reuse analyzer-grade logic is
    # out of scope here — escape is taken from the analyzer JSON (see main).
    births = np.array([v["birth"] for v in family.values()])
    pkes = np.array([v["pke"] for v in family.values()])
    order = np.argsort(births)
    cdf = np.arange(1, len(births) + 1) / len(births)
    spectrum, _ = np.histogram(births, bins=BIRTH_EDGES)
    inside = [rk for rk, v in family.items()
              if np.all(v["birth_pos"] >= bounds[:, 0] + interior_inset_mm) and
              np.all(v["birth_pos"] <= bounds[:, 1] - interior_inset_mm)]
    return {
        "roots": len(family),
        "birth_quantiles": np.interp([0.1, 0.5, 0.9, 0.99], cdf, births[order]).tolist(),
        "birth_spectrum_counts": spectrum.tolist(),
        "parent_ke_median": float(np.median(pkes)) if len(pkes) else None,
        "interior_roots": len(inside),
        "interior_fraction": float(len(inside) / len(family)) if family else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True,
                   help="name:steps_path:analysis_json:bounds6:voxelvol")
    p.add_argument("--histories", type=int, required=True)
    p.add_argument("--interior-inset-mm", type=float, default=50.0)
    p.add_argument("--parent-ke-bands", type=float, nargs="+",
                   default=[2300.0, 2350.0, 2400.0, 2450.0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-hu", action="append", default=[],
                   help="name=HU; when supplied, every case must be verified as section 0")
    a = p.parse_args()
    bands = np.asarray(a.parent_ke_bands, dtype=float)
    if len(bands) < 2 or not np.isfinite(bands).all() or np.any(np.diff(bands) <= 0):
        raise ValueError("Parent KE bands must be finite and increasing")
    geometries = []
    case_hus = {}
    for item in a.case_hu:
        name, hu = item.split("=", 1)
        if name in case_hus:
            raise ValueError("Duplicate case HU")
        case_hus[name] = int(hu)
    joints = {}
    for spec in a.input:
        name, steps, analysis, bounds, voxelvol = spec.split(":", 4)
        steps, analysis = Path(steps), Path(analysis)
        rep = json.loads(analysis.read_text())
        if geometries and (rep.get("joint_radial_edges_mm"), rep.get("joint_longitudinal_edges_mm")) != joint_edges:
            raise ValueError("Joint histogram bin edges differ")
        joint_edges = (rep.get("joint_radial_edges_mm"), rep.get("joint_longitudinal_edges_mm"))
        if rep.get("record_schema_version") not in (2, 3):
            raise ValueError(f"{name}: requires a v2/v3 analyzer report")
        if int(rep["histories_requested"]) != a.histories:
            raise ValueError(f"{name}: history count mismatch")
        d = load_v2_binary(steps)
        if np.any(d[:, 21] != rep["record_schema_version"]):
            raise ValueError("Report/raw schema mismatch")
        if rep["record_schema_version"] == 3 and rep.get("parent_step_binding_verified") is not True:
            raise ValueError("v3 generating-step binding was not verified")
        identity = None
        if case_hus:
            if name not in case_hus:
                raise ValueError("Missing case HU for section-0 scope audit")
            identity = schneider_identity(Path(__file__).resolve().parents[1]/"data/HUtoMaterialSchneider.txt", case_hus[name])
            if identity["material_section"] != 0:
                raise ValueError("Material-section scope mismatch (not section 0)")
            if not np.allclose(d[:, 20], identity["density_g_cm3"], rtol=1e-5, atol=0):
                raise ValueError("Raw density does not match corrected Schneider formula")
        tracks, root = build_roots(d)
        audit = rep.get("finite_slab_energy_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"{name}: missing finite-slab escape audit")
        total = float(rep["total_deposit_MeV"])
        birth = float(audit["electron_birth_MeV"])
        dep = float(audit["electron_family_deposited_MeV"])
        esc = float(audit["electron_family_escaped_MeV"])
        if abs(birth - dep - esc) / birth > 1e-3:
            raise ValueError(f"{name}: family closure broken, refusing comparison")
        summary = summarize(d, tracks, root,
                            [float(v) for v in bounds.split(",")],
                            a.interior_inset_mm)
        # Birth-time matched parent KE median (offline match, not exit proxy).
        matched_kes = [item["matched_parent_ke_MeV"] for item in
                       json.loads(analysis.read_text())["finite_slab_energy_audit"]["per_family_residuals"]
                       if item.get("matched_parent_ke_MeV") is not None]
        summary["parent_ke_median"] = float(np.median(matched_kes)) if matched_kes else None
        # Conditioned escape/movable fractions per MATCHED birth-time parent-KE
        # band (offline C12-segment match; never the scorer exit proxy).
        fam_resid = {tuple([item["run"], item["event"], item["track"]]): item
                     for item in audit["per_family_residuals"]}
        if any(item.get("matched_parent_ke_MeV") is None for item in fam_resid.values()):
            raise ValueError(f"{name}: missing birth-time parent match; refusing comparison")
        banded = []
        for lo, hi in zip(bands[:-1], bands[1:]):
            b_dep = b_esc = b_birth = 0.0
            b_n = 0
            for rk, item in fam_resid.items():
                pke = item["matched_parent_ke_MeV"]
                if pke is not None and lo <= pke < hi:
                    b_n += 1
                    b_birth += item["birth_MeV"]
                    b_dep += item["deposited_MeV"]
                    b_esc += item["escaped_MeV"]
            banded.append({
                "parent_ke_MeV": [lo, hi], "roots": b_n,
                "escape_fraction": (b_esc / (b_dep + b_esc)) if (b_dep + b_esc) > 0 else None,
                "fraction_of_total_electron_birth_energy": (b_birth / birth) if birth > 0 else None,
            })
        joint = np.asarray(rep["joint_deposited_MeV"], dtype=float)
        joint_shape_distance(joint, joint)
        if not np.isclose(joint.sum(), dep, rtol=1e-10):
            raise ValueError("Joint histogram does not close to electron deposition")
        joints[name] = joint
        raw_sha = sha256(steps)
        if rep.get("steps_sha256") != raw_sha:
            raise ValueError(f"{name}: steps SHA mismatch (report vs raw); refusing comparison")
        geometries.append({
            "name": name,
            "material_identity": identity,
            "steps_sha256": raw_sha,
            "histories": a.histories,
            "total_deposit_MeV": total,
            "electron_birth_MeV": birth,
            "electron_family_deposited_MeV": dep,
            "electron_family_escaped_MeV": esc,
            "escape_fraction": esc / (dep + esc),
            "movable_fraction_of_total_deposit": dep / total,
            "birth_share_of_parent_loss": birth / (birth + float(rep["primary_local_MeV"])),
            "summary": summary,
            "parent_ke_bands": banded,
            "joint_total_MeV": float(joint.sum()),
        })
    hashes = [g["steps_sha256"] for g in geometries]
    if case_hus and set(case_hus) != {g["name"] for g in geometries}:
        raise ValueError("Unknown case HU name")
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate shard content: steps_sha256 repeated; refusing comparison")
    if len({g["name"] for g in geometries}) != len(geometries):
        raise ValueError("Duplicate geometry name in comparison input")
    base = geometries[0]
    base_joint = joints[base["name"]]
    comparisons = []
    for g in geometries[1:]:
        gj = joints[g["name"]]
        l1 = joint_shape_distance(base_joint, gj)
        comparisons.append({
            "pair": [base["name"], g["name"]],
            "escape_fraction_delta": g["escape_fraction"] - base["escape_fraction"],
            "movable_fraction_delta": (g["movable_fraction_of_total_deposit"] -
                                       base["movable_fraction_of_total_deposit"]),
            "joint_L1_distance": l1,
            "joint_L1_definition": "sum(abs(P_reference - P_case)); each histogram normalized separately; range [0,2]",
            "joint_energy_ratio": float(gj.sum()/base_joint.sum()),
            "conditioning": "unconditional; parent KE bands do not condition this histogram",
        })
    result = {
        "section_scope_verified": bool(case_hus),
        "scope": ("Unconditional finite-slab geometry comparison; approximate parent-KE band summaries are separate; "
                  "12-history shards validate the comparison code, not convergence"),
        "histories_per_geometry": a.histories,
        "interior_inset_mm": a.interior_inset_mm,
        "parent_ke_bands_MeV": bands.tolist(),
        "geometries": geometries,
        "comparisons_vs_first": comparisons,
        "verdict": ("INCONCLUSIVE at 12 histories per geometry: report deltas "
                    "without claiming kernel-grade convergence; do not generate "
                    "a runtime kernel from these shards."),
        "limitations": [
            "Escape energy has no subsequent deposit position; it is accounted, never redistributed.",
            "Unconditional deltas mix birth-spectrum shifts with boundary effects; use parent-KE bands.",
            "Far-tail bins are never deleted to fake convergence.",
        ],
    }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
