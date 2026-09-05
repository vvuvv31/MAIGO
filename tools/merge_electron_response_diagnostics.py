#!/usr/bin/env python3
"""Aggregate per-shard electron-response analyses without faking statistics.

Merges raw weighted joint histograms, birth/deposit/escape energies and their
denominators first; probabilities are recomputed from the merged histogram.
Shard quantiles are never averaged: pooled longitudinal quantiles are left as
null with an explicit reason because exact pooled quantiles require pooled raw
offsets, not averaged shard quantiles. Electron steps are never counted as
independent histories; history counts sum over shards.

Hard failures (non-zero exit, no PASS report):
  order must not matter; duplicate shards, missing/failed shards, SHA mismatch
  and mixed configs are all rejected.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_report(path):
    with open(path) as stream:
        return json.load(stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True,
                   help="Per-shard analyzer JSON reports (order-independent)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-shards", type=int, default=None,
                   help="Fail unless exactly this many unique shards merge")
    p.add_argument("--expected-config-sha", type=str, default=None,
                   help="Fail if shards were produced under mixed configs")
    p.add_argument("--verify-raw", action="store_true",
                   help="Recompute steps/header/dose SHA256 from raw paths in reports")
    p.add_argument("--disk-budget-gib", type=float, default=None,
                   help="Fail if total raw bytes exceed this budget")
    a = p.parse_args()
    ordered = sorted(str(r) for r in a.reports)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Duplicate report path in merge input")
    reports = [load_report(Path(r)) for r in ordered]
    if not reports:
        raise ValueError("No shard reports provided")
    if a.expected_shards is not None and len(reports) != a.expected_shards:
        raise ValueError(
            f"Missing or extra shards: expected {a.expected_shards}, got {len(reports)}; "
            f"refusing to emit a complete-campaign PASS")
    # Duplicate-shard detection: same raw content must not be counted twice.
    step_hashes = [r.get("steps_sha256") for r in reports]
    if len(set(step_hashes)) != len(step_hashes):
        raise ValueError("Duplicate shard content: steps_sha256 repeated; refusing merge")
    # Config mixing: analyzer reports do not carry the TOPAS config hash, so the
    # caller passes per-shard metadata sidecars when strict mixing checks apply.
    # Here we enforce identical bin edges and input format at minimum, and an
    # explicit expected-config gate when the caller knows the campaign config.
    first = reports[0]
    radial_edges = first.get("joint_radial_edges_mm")
    longitudinal_edges = first.get("joint_longitudinal_edges_mm")
    input_format = first.get("input_format")
    for edges in (radial_edges, longitudinal_edges):
        v = np.asarray(edges, dtype=float)
        if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all() or np.any(np.diff(v) <= 0):
            raise ValueError("Invalid histogram bin edges")
    for index, rep in enumerate(reports):
        if rep.get("record_schema_version") != first.get("record_schema_version"):
            raise ValueError("Mixed record schemas")
        if rep.get("joint_radial_edges_mm") != radial_edges:
            raise ValueError(f"Shard {index} radial bin edges differ; refusing mixed merge")
        if rep.get("joint_longitudinal_edges_mm") != longitudinal_edges:
            raise ValueError(f"Shard {index} longitudinal bin edges differ; refusing mixed merge")
        if rep.get("input_format") != input_format:
            raise ValueError(f"Shard {index} input format differs; refusing mixed merge")
        for field in ("total_deposit_MeV", "delta_family_deposit_MeV",
                      "joint_deposited_MeV", "histories_requested"):
            if field not in rep or rep[field] is None:
                raise ValueError(f"Shard {index} missing required field {field}; "
                                 f"failed shards cannot merge as complete")
    if a.expected_config_sha is not None:
        sidecars = []
        for raw in ordered:
            meta = Path(raw).parent / "metadata.json"
            if meta.is_file():
                try:
                    sidecars.append(json.load(open(meta)).get("config_sha256"))
                except (OSError, ValueError):
                    sidecars.append(None)
            else:
                sidecars.append(None)
        if any(s is None for s in sidecars):
            raise ValueError("Missing metadata/config SHA: strict config verification requires every shard")
        if any(s != a.expected_config_sha for s in sidecars):
            raise ValueError("Shard config SHA mismatch: refusing to mix configurations")
    if a.verify_raw:
        for index, (raw, rep) in enumerate(zip(ordered, reports)):
            for kind in ("steps", "header", "dose"):
                field = kind + "_sha256"
                path = rep.get("raw_" + kind)
                if not rep.get(field) or not path:
                    raise ValueError(f"Shard {index} missing raw path or recorded {field}; re-analyze legacy report")
                if sha256_file(Path(path)) != rep[field]:
                    raise ValueError(f"Shard {index} raw {kind} SHA mismatch")
    total_raw_bytes = 0
    if a.disk_budget_gib is not None:
        raw_paths = set()
        for rep in reports:
            for kind in ("steps", "header", "dose"):
                if not rep.get("raw_" + kind):
                    raise ValueError("Disk audit requires explicit raw paths; re-analyze legacy report")
                raw_paths.add(Path(rep["raw_" + kind]).resolve())
        total_raw_bytes = sum(path.stat().st_size for path in raw_paths)
        budget = int(a.disk_budget_gib * 1024**3)
        if total_raw_bytes > budget:
            raise ValueError(
                f"Raw bytes {total_raw_bytes} exceed disk budget {budget}; "
                f"stop submitting new shards, keep completed data")
    # Exact pooled sums (histories are independent across shards with unique
    # seeds; steps within a shard are correlated and never counted as histories).
    joint_sum = None
    pooled = {
        "histories_requested": 0,
        "events_observed": 0,
        "rows": 0,
        "tracks": 0,
        "total_deposit_MeV": 0.0,
        "primary_local_MeV": 0.0,
        "delta_family_deposit_MeV": 0.0,
        "dose3d_integrated_MeV": 0.0,
    }
    per_shard = []
    for raw, rep in zip(ordered, reports):
        if rep["histories_requested"] <= 0 or rep["events_observed"] != rep["histories_requested"]:
            raise ValueError("Incomplete shard event coverage")
        for field in ("total_deposit_MeV", "primary_local_MeV", "delta_family_deposit_MeV", "dose3d_integrated_MeV"):
            if not np.isfinite(rep[field]) or rep[field] < 0:
                raise ValueError(f"Invalid shard energy {field}")
        block = np.asarray(rep["joint_deposited_MeV"], dtype=float)
        if (block.shape != (len(radial_edges)-1, len(longitudinal_edges)-1)
                or not np.isfinite(block).all() or np.any(block < 0)):
            raise ValueError("Invalid shard histogram shape/values")
        if not np.isclose(block.sum(), rep["delta_family_deposit_MeV"], rtol=1e-10, atol=1e-12):
            raise ValueError("Shard joint energy mismatch (cannot cancel across shards)")
        if rep["total_deposit_MeV"] <= 0 or block.sum() <= 0:
            raise ValueError("Zero shard energy")
        if abs(rep["dose3d_integrated_MeV"]/rep["total_deposit_MeV"]-1) > 1e-3:
            raise ValueError("Shard recomputed dose closure failed")
        pooled["histories_requested"] += int(rep["histories_requested"])
        pooled["events_observed"] += int(rep["events_observed"])
        pooled["rows"] += int(rep["rows"])
        pooled["tracks"] += int(rep["tracks"])
        pooled["total_deposit_MeV"] += float(rep["total_deposit_MeV"])
        pooled["primary_local_MeV"] += float(rep["primary_local_MeV"])
        pooled["delta_family_deposit_MeV"] += float(rep["delta_family_deposit_MeV"])
        pooled["dose3d_integrated_MeV"] += float(rep["dose3d_integrated_MeV"])
        block = np.asarray(rep["joint_deposited_MeV"], dtype=float)
        joint_sum = block.copy() if joint_sum is None else joint_sum + block
        dose_ratio = float(rep["dose3d_over_steps"])
        if abs(dose_ratio - 1) > 1e-3:
            raise ValueError(f"Shard {raw} dose closure {dose_ratio} failed; refusing merge")
        per_shard.append({
            "report": raw,
            "histories_requested": int(rep["histories_requested"]),
            "total_deposit_MeV": float(rep["total_deposit_MeV"]),
            "delta_family_deposit_MeV": float(rep["delta_family_deposit_MeV"]),
            "steps_sha256": rep.get("steps_sha256"),
            "analyzer_sha256": rep.get("analyzer_sha256"),
        })
    joint_total = float(joint_sum.sum())
    if not np.isclose(joint_total, pooled["delta_family_deposit_MeV"], rtol=1e-10):
        raise ValueError(
            f"Pooled joint {joint_total} != pooled family deposit "
            f"{pooled['delta_family_deposit_MeV']}: histogram accounting broken")
    # Finite-slab escape audits merge by summation when every shard provides one.
    audits = [r.get("finite_slab_energy_audit") for r in reports]
    pooled_audit = None
    if all(isinstance(item, dict) for item in audits):
        pooled_audit = {
            "incident_MeV": float(sum(item["incident_MeV"] for item in audits)),
            "outgoing_MeV": float(sum(item["outgoing_MeV"] for item in audits)),
            "primary_outgoing_MeV": float(sum(item["primary_outgoing_MeV"] for item in audits)),
            "electron_root_count": int(sum(item["electron_root_count"] for item in audits)),
            "escaped_tracks": int(sum(item["escaped_tracks"] for item in audits)),
            "electron_birth_MeV": float(sum(item["electron_birth_MeV"] for item in audits)),
            "electron_family_deposited_MeV": float(
                sum(item["electron_family_deposited_MeV"] for item in audits)),
            "electron_family_escaped_MeV": float(
                sum(item["electron_family_escaped_MeV"] for item in audits)),
        }
        inc = pooled_audit["incident_MeV"]
        birth = pooled_audit["electron_birth_MeV"]
        pooled_audit["global_relative_residual"] = float(
            (inc - pooled["total_deposit_MeV"] - pooled_audit["outgoing_MeV"]) / inc)
        pooled_audit["electron_relative_residual"] = float(
            (birth - pooled_audit["electron_family_deposited_MeV"]
             - pooled_audit["electron_family_escaped_MeV"]) / birth)
    merged = {
        "raw_verified": bool(a.verify_raw),
        "config_verified": a.expected_config_sha is not None,
        "campaign": "electron_response_shards",
        "shard_count": len(reports),
        "shard_reports_in_sorted_order": ordered,
        "histories_requested": pooled["histories_requested"],
        "events_observed": pooled["events_observed"],
        "rows": pooled["rows"],
        "tracks": pooled["tracks"],
        "total_deposit_MeV": pooled["total_deposit_MeV"],
        "primary_local_MeV": pooled["primary_local_MeV"],
        "delta_family_deposit_MeV": pooled["delta_family_deposit_MeV"],
        "delta_fraction_of_deposit": float(
            pooled["delta_family_deposit_MeV"] / pooled["total_deposit_MeV"]),
        "dose3d_integrated_MeV": pooled["dose3d_integrated_MeV"],
        "dose3d_over_steps": float(
            pooled["dose3d_integrated_MeV"] / pooled["total_deposit_MeV"]),
        "joint_radial_edges_mm": radial_edges,
        "joint_longitudinal_edges_mm": longitudinal_edges,
        "joint_deposited_MeV": joint_sum.tolist(),
        "joint_probability": (joint_sum / joint_sum.sum()).tolist(),
        "pooled_longitudinal_quantiles_mm": None,
        "pooled_longitudinal_quantiles_status": (
            "not computed: exact pooled quantiles require pooled raw offsets; "
            "shard quantiles must not be averaged"),
        "per_shard_longitudinal_mid_mm": [
            {"report": raw, "values": rep.get("longitudinal_mid_mm")} for raw, rep in zip(ordered, reports)],
        "finite_slab_energy_audit_pooled": pooled_audit,
        "per_shard": per_shard,
        "scope": "Pooled homogeneous diagnostic shards; not a compiled transport kernel",
        "limitations": [
            "Two small shards validate aggregation code only; they do not prove statistical convergence.",
            "Pooled quantiles intentionally null; averaging shard quantiles is forbidden.",
            "Steps are correlated within histories and are never counted as independent histories.",
        ],
    }
    payload = json.dumps(merged, indent=2, allow_nan=False)
    Path(a.output).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
