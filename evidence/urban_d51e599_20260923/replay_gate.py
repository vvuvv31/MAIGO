#!/usr/bin/env python3
"""Identity-checked phase-space replay acceptance gate.

The gate reads every required interval and threshold from acceptance.yaml.
It needs an explicit source-record -> (RunID, EventID, TrackID) map and a
GPU metadata sidecar. Missing provenance or samples produce BLOCKED (exit 2),
measured threshold failures produce FAIL (exit 1), and only a complete pass
returns 0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    sw = float(weights.sum())
    require(sw > 0.0 and np.isfinite(sw), "non-positive/non-finite sample weight")
    return float(np.dot(values, weights) / sw)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    require(total > 0.0, "empty weighted quantile")
    index = int(np.searchsorted(np.cumsum(w), q * total, side="left"))
    return float(v[min(index, len(v) - 1)])


def load_accept(path: Path) -> tuple[dict, list[tuple[float, float]]]:
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))
        acc = root["conditional_phase_space"]
        intervals = [(float(a), float(b)) for a, b in acc["intervals_mm"]]
    except Exception as exc:
        raise GateError(f"cannot parse conditional_phase_space acceptance: {exc}")
    require(bool(intervals), "acceptance.yaml has no required intervals")
    require(int(acc["seeds_batches"]) >= 2, "seeds_batches must be >= 2")
    for key in ("angvar_rel", "disvar_rel", "q999_rel", "tail_prob_abs",
                "survival_rel"):
        require(key in acc and math.isfinite(float(acc[key])) and
                float(acc[key]) >= 0.0, f"missing/invalid gate {key}")
    return acc, intervals


def load_identity_map(path: Path) -> tuple[dict[int, tuple[tuple[int, int, int], float]], dict]:
    required = {"source_history", "source_record_id", "run_id", "event_id",
                "track_id", "weight"}
    by_history: dict[int, tuple[tuple[int, int, int], float]] = {}
    by_topas: dict[tuple[int, int, int], int] = {}
    source_records: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                f"identity map must contain {sorted(required)}")
        for line_no, row in enumerate(reader, 2):
            try:
                sid = int(row["source_history"])
                source_record = row["source_record_id"].strip()
                key = (int(row["run_id"]), int(row["event_id"]),
                       int(row["track_id"]))
                weight = float(row["weight"])
            except Exception as exc:
                raise GateError(f"invalid identity row {line_no}: {exc}")
            require(sid >= 0 and bool(source_record) and math.isfinite(weight) and
                    weight > 0.0, f"invalid identity/weight at row {line_no}")
            require(sid not in by_history, f"duplicate source_history {sid}")
            require(key not in by_topas, f"duplicate TOPAS identity {key}")
            require(source_record not in source_records,
                    f"duplicate source_record_id {source_record}")
            by_history[sid] = (key, weight)
            by_topas[key] = sid
            source_records.add(source_record)
    require(bool(by_history), "identity map has no source records")
    require(set(by_history) == set(range(len(by_history))),
            "source_history must be a contiguous zero-based source-record index")
    return by_history, by_topas


def validate_gpu_metadata(path: Path, gpu_path: Path, identity_path: Path,
                          nsource: int) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"cannot read GPU metadata JSON: {exc}")
    required = ("gpu_csv_sha256", "identity_map_sha256", "source_record_count",
                "source_records_file", "source_records_sha256", "config_file",
                "config_sha256", "coordinate_map",
                "particle_z", "particle_a", "weight_semantics")
    missing = [key for key in required if key not in meta]
    require(not missing, f"GPU metadata missing {missing}")
    require(meta["gpu_csv_sha256"] == sha256(gpu_path),
            "GPU CSV hash does not match metadata")
    require(meta["identity_map_sha256"] == sha256(identity_path),
            "identity-map hash does not match GPU metadata")
    source_path = Path(meta["source_records_file"])
    config_path = Path(meta["config_file"])
    require(source_path.is_file() and sha256(source_path) == meta["source_records_sha256"],
            "source-record file is missing or its hash differs from metadata")
    require(config_path.is_file() and sha256(config_path) == meta["config_sha256"],
            "effective GPU config is missing or its hash differs from metadata")
    require(int(meta["source_record_count"]) == nsource,
            "GPU source-record count does not match identity map")
    require(meta["particle_z"] == 6 and meta["particle_a"] == 12,
            "replay metadata is not C12")
    require(meta["weight_semantics"] == "per-source-particle statistical weight",
            "unsupported GPU weight semantics")
    coords = meta["coordinate_map"]
    require(coords == {"position": ["x_mm", "y_mm", "depth_mm"],
                       "direction": ["direction_x", "direction_y", "direction_z"],
                       "topas_to_gpu_position": ["X", "Z", "Y"],
                       "topas_to_gpu_direction": ["dirX", "dirZ_inferred", "dirY"]},
            "coordinate/axis map is absent or differs from the audited convention")
    return meta


def validate_topas_metadata(path: Path, identity_path: Path,
                            source_sha: str, intervals: list[tuple[float, float]]) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"cannot read TOPAS metadata JSON: {exc}")
    required = ("config_file", "config_sha256", "source_records_file",
                "source_records_sha256", "identity_map_sha256",
                "coordinate_map", "weight_semantics")
    missing = [key for key in required if key not in meta]
    require(not missing, f"TOPAS metadata missing {missing}")
    config, source = Path(meta["config_file"]), Path(meta["source_records_file"])
    require(config.is_file() and sha256(config) == meta["config_sha256"],
            "effective TOPAS config is missing or its hash differs from metadata")
    require(source.is_file() and sha256(source) == source_sha == meta["source_records_sha256"],
            "TOPAS source-record file/hash does not match GPU source")
    require(meta["identity_map_sha256"] == sha256(identity_path),
            "TOPAS metadata identity-map hash differs")
    require(meta["weight_semantics"] == "per-source-particle statistical weight",
            "unsupported TOPAS weight semantics")
    require(meta["coordinate_map"] == {
        "position_units": "cm", "direction": ["dirX", "dirZ_inferred", "dirY"],
        "depth_axis": "TOPAS_Y", "transverse_axes": ["TOPAS_X", "TOPAS_Z"]},
        "TOPAS coordinate/unit map is absent or differs from the audited convention")
    require(meta.get("required_intervals_mm") == [[a, b] for a, b in intervals],
            "TOPAS metadata required intervals differ from acceptance.yaml")
    return meta


def validate_topas_header(path: Path, expected_records: int | None = None) -> dict:
    require(path.is_file(), f"missing TOPAS phase-space header: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    require("Position X [cm]" in raw and "Position Y [cm]" in raw and
            "Position Z [cm]" in raw and "Direction Cosine X" in raw and
            "Direction Cosine Y" in raw and "Weight" in raw and
            "Run ID" in raw and "Event ID" in raw and "Track ID" in raw,
            f"TOPAS header lacks required units or identity fields: {path}")
    records = None
    for line in raw.splitlines():
        if "Number of Scored Particles:" in line:
            try:
                records = int(line.rsplit(":", 1)[1].strip())
            except ValueError:
                pass
    if expected_records is not None and records is not None:
        require(records == expected_records,
                f"TOPAS header scored count mismatch: {path}")
    return {"path": str(path), "sha256": sha256(path),
            "scored_particles": records}


def load_gpu(path: Path, by_history: dict, intervals: list[tuple[float, float]]) -> dict[int, dict]:
    with path.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    fields = ("source_history", "plane_index", "depth_mm", "kinetic_energy_MeV",
              "x_mm", "y_mm", "direction_x", "direction_y", "direction_z", "weight",
              "atomic_number", "mass_number")
    require(set(fields).issubset(header), f"GPU CSV lacks required columns {fields}")
    try:
        rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64,
                          usecols=tuple(header.index(k) for k in fields), ndmin=2)
    except Exception as exc:
        raise GateError(f"cannot load numeric GPU plane CSV: {exc}")
    require(rows.size > 0 and np.isfinite(rows).all(),
            "GPU plane file is empty or contains NaN/Inf")
    ix = {name: i for i, name in enumerate(fields)}
    out: dict[int, dict] = {i: {} for i in range(len(intervals) + 1)}
    seen = set()
    expected_depths = [intervals[0][0]] + [end for _, end in intervals]
    for r in rows:
        sid_f, plane_f = r[ix["source_history"]], r[ix["plane_index"]]
        sid, plane = int(sid_f), int(plane_f)
        require(sid_f == sid and plane_f == plane,
                "GPU identity/plane columns must be exact integers")
        require(sid in by_history, f"GPU source_history {sid} is absent from identity map")
        require(0 <= plane < len(expected_depths), f"unexpected GPU plane index {plane}")
        key = (sid, plane)
        require(key not in seen, f"duplicate GPU source/plane record {key}")
        seen.add(key)
        require(int(r[ix["atomic_number"]]) == 6 and
                int(r[ix["mass_number"]]) == 12,
                "non-C12 record in primary replay output")
        require(abs(r[ix["depth_mm"]] - expected_depths[plane]) < 0.1,
                f"GPU plane {plane} depth/axis mismatch")
        w = float(r[ix["weight"]])
        require(abs(w - by_history[sid][1]) <= 1.0e-6 * max(w, 1.0),
                f"GPU weight differs from source-record weight for {sid}")
        d = r[[ix["direction_x"], ix["direction_y"], ix["direction_z"]]]
        norm = float(np.linalg.norm(d))
        require(abs(norm - 1.0) <= 1.0e-4,
                f"GPU direction is not normalized for source {sid}")
        e = float(r[ix["kinetic_energy_MeV"]])
        require(e >= 0.0, f"negative GPU energy for source {sid}")
        out[plane][sid] = np.array([
            r[ix["x_mm"]], r[ix["y_mm"]], r[ix["depth_mm"]],
            *d, e, w], dtype=np.float64)
    return out


def load_topas(path: Path, expected_depth_mm: float,
               by_topas: dict[tuple[int, int, int], int],
               by_history: dict[int, tuple[tuple[int, int, int], float]]) -> dict[int, np.ndarray]:
    require(path.is_file(), f"missing TOPAS plane file: {path}")
    try:
        data = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    except Exception as exc:
        raise GateError(f"cannot parse TOPAS phase space {path}: {exc}")
    require(data.shape[0] > 0 and data.shape[1] == 14 and
            np.isfinite(data).all(), f"empty, malformed, or non-finite TOPAS file: {path}")
    out = {}
    seen = set()
    for row in data:
        pdg, parent = int(row[7]), int(row[13])
        if pdg != 1000060120 or parent != 0:
            continue
        run, event, track = int(row[10]), int(row[11]), int(row[12])
        key = (run, event, track)
        require(key in by_topas, f"unmapped TOPAS C12 primary identity {key} in {path.name}")
        sid = by_topas[key]
        require(sid not in seen, f"duplicate TOPAS source record {sid} in {path.name}")
        seen.add(sid)
        x_mm, depth_mm, y_mm = row[0] * 10.0, row[1] * 10.0, row[2] * 10.0
        require(abs(depth_mm - (60.0 + expected_depth_mm)) < 0.1,
                f"TOPAS plane depth/beam-axis mismatch in {path.name}")
        dx, axial = row[3], row[4]
        radial2 = 1.0 - dx * dx - axial * axial
        require(radial2 >= -1.0e-8,
                f"illegal TOPAS direction cosines for source {sid}")
        dy = math.sqrt(max(0.0, radial2))
        if row[8] > 0.5:
            dy = -dy
        direction = np.array([dx, dy, axial], dtype=np.float64)
        require(abs(float(np.linalg.norm(direction)) - 1.0) <= 1.0e-4,
                f"TOPAS direction is not normalized for source {sid}")
        weight = float(row[6])
        require(weight > 0.0 and abs(weight - by_history[sid][1]) <=
                1.0e-6 * max(weight, 1.0),
                f"TOPAS weight differs from source-record weight for {sid}")
        out[sid] = np.array([x_mm, y_mm, depth_mm, *direction,
                             row[5], weight], dtype=np.float64)
    return out


def _moments(rows: list[np.ndarray], ids: np.ndarray, batches: int) -> dict:
    if len(ids) == 0:
        return {}
    require(len(rows) == len(ids), "pair rows and source IDs differ in length")
    v = np.stack(rows)
    x0, y0, z0 = v[:, 0, 0], v[:, 0, 1], v[:, 0, 2]
    x1, y1, z1 = v[:, 1, 0], v[:, 1, 1], v[:, 1, 2]
    u0, u1 = v[:, 0, 3:6], v[:, 1, 3:6]
    cross = np.linalg.norm(np.cross(u0, u1), axis=1)
    dot = np.clip(np.einsum("ij,ij->i", u0, u1), -1.0, 1.0)
    theta = np.arctan2(cross, dot) * 1000.0
    raw_x, raw_y = x1 - x0, y1 - y0
    axial_ok = u0[:, 2] > 1.0e-4
    dz = z1 - z0
    rx = raw_x[axial_ok] - u0[axial_ok, 0] / u0[axial_ok, 2] * dz[axial_ok]
    ry = raw_y[axial_ok] - u0[axial_ok, 1] / u0[axial_ok, 2] * dz[axial_ok]
    w = v[:, 0, 7]  # row schema: x,y,z,ux,uy,uz,kinetic_energy,weight
    wr = w[axial_ok]
    if len(rx) == 0:
        return {}
    wt = weighted_quantile(theta, w, 0.999)
    out = {
        "n": int(len(ids)), "weight_sum": float(w.sum()),
        "theta2": weighted_mean(theta * theta, w),
        "theta_q999_mrad": wt,
        "theta_gt100mrad": weighted_mean((theta > 100.0).astype(float), w),
        "raw_displacement2_mm2": weighted_mean(raw_x**2 + raw_y**2, w),
        "scatter_residual2_mm2": weighted_mean(rx**2 + ry**2, wr),
        "residual_n": int(len(rx)), "near_transverse_or_reverse_n": int(len(ids)-len(rx)),
        "cov_residual_x_dtheta_x": float(np.average(
            (rx - np.average(rx, weights=wr)) *
            (u1[axial_ok, 0] - u0[axial_ok, 0] -
             np.average(u1[axial_ok, 0] - u0[axial_ok, 0], weights=wr)),
            weights=wr)),
        "cov_residual_y_dtheta_y": float(np.average(
            (ry - np.average(ry, weights=wr)) *
            (u1[axial_ok, 1] - u0[axial_ok, 1] -
             np.average(u1[axial_ok, 1] - u0[axial_ok, 1], weights=wr)),
            weights=wr)),
    }
    bstats = {key: [] for key in ("theta2", "theta_q999_mrad", "theta_gt100mrad",
                                   "raw_displacement2_mm2", "scatter_residual2_mm2")}
    bid = ids % batches
    for b in range(batches):
        take = bid == b
        if np.count_nonzero(take) < 2:
            continue
        vv, ww = v[take], w[take]
        uu0, uu1 = vv[:, 0, 3:6], vv[:, 1, 3:6]
        th = np.arctan2(np.linalg.norm(np.cross(uu0, uu1), axis=1),
                       np.clip(np.einsum("ij,ij->i", uu0, uu1), -1.0, 1.0)) * 1000.0
        dxr = vv[:, 1, 0] - vv[:, 0, 0]
        dyr = vv[:, 1, 1] - vv[:, 0, 1]
        bstats["theta2"].append(weighted_mean(th**2, ww))
        bstats["theta_q999_mrad"].append(weighted_quantile(th, ww, 0.999))
        bstats["theta_gt100mrad"].append(weighted_mean((th > 100).astype(float), ww))
        bstats["raw_displacement2_mm2"].append(weighted_mean(dxr**2 + dyr**2, ww))
        ok = uu0[:, 2] > 1.0e-4
        if np.count_nonzero(ok) >= 2:
            rxb = dxr[ok] - uu0[ok, 0] / uu0[ok, 2] * (vv[ok, 1, 2]-vv[ok, 0, 2])
            ryb = dyr[ok] - uu0[ok, 1] / uu0[ok, 2] * (vv[ok, 1, 2]-vv[ok, 0, 2])
            bstats["scatter_residual2_mm2"].append(weighted_mean(rxb**2 + ryb**2, ww[ok]))
    out["batch_means"] = bstats
    out["batch_count_required"] = batches
    return out


def _survival(plane0: dict, plane1: dict, by_history: dict, batches: int) -> dict:
    ids = np.array(sorted(plane0), dtype=np.int64)
    if len(ids) == 0:
        return {}
    reached = np.array([int(i in plane1) for i in ids], dtype=np.float64)
    w = np.array([by_history[int(i)][1] for i in ids])
    p = weighted_mean(reached, w)
    means = []
    for b in range(batches):
        take = ids % batches == b
        if np.count_nonzero(take):
            means.append(weighted_mean(reached[take], w[take]))
    se = float(np.std(means, ddof=1) / math.sqrt(len(means))) if len(means) == batches else None
    return {"start_n": int(len(ids)), "end_n": int(sum(i in plane1 for i in ids)),
            "survival_fraction": p, "batch_se": se,
            "batch_fractions": means, "batch_count_required": batches}


def _ratio_gate(gpu: dict, ref: dict, metric: str, limit: float,
                batches_required: int) -> dict:
    a, b = float(gpu[metric]), float(ref[metric])
    if not (math.isfinite(a) and math.isfinite(b) and b > 0.0):
        return {"status": "INCONCLUSIVE", "ratio": None, "se": None}
    ratio = a / b
    ga, rb = gpu["batch_means"][metric], ref["batch_means"][metric]
    if len(ga) != batches_required or len(rb) != batches_required:
        return {"status": "INCONCLUSIVE", "ratio": ratio, "se": None}
    if not all(math.isfinite(ga[i]) and math.isfinite(rb[i]) and rb[i] > 0.0
               for i in range(batches_required)):
        return {"status": "INCONCLUSIVE", "ratio": ratio, "se": None}
    batch_ratios = np.asarray(ga) / np.asarray(rb)
    se = float(np.std(batch_ratios, ddof=1) / math.sqrt(batches_required))
    passed = abs(ratio - 1.0) + 2.0 * se < limit
    return {"status": "PASS" if passed else "FAIL", "ratio": ratio,
            "se": se, "gate": limit, "equivalence_margin": abs(ratio-1.0)+2*se}


def _absolute_gate(gpu: dict, ref: dict, metric: str, limit: float,
                   batches_required: int) -> dict:
    diff = float(gpu[metric]) - float(ref[metric])
    ga, rb = gpu["batch_means"][metric], ref["batch_means"][metric]
    if len(ga) != batches_required or len(rb) != batches_required:
        return {"status": "INCONCLUSIVE", "difference": diff, "se": None}
    diffs = np.asarray(ga) - np.asarray(rb)
    if not np.isfinite(diffs).all():
        return {"status": "INCONCLUSIVE", "difference": diff, "se": None}
    se = float(np.std(diffs, ddof=1) / math.sqrt(batches_required))
    passed = abs(diff) + 2.0 * se < limit
    return {"status": "PASS" if passed else "FAIL", "difference": diff,
            "se": se, "gate": limit, "equivalence_margin": abs(diff)+2*se}


def run(args) -> tuple[dict, int]:
    acc, intervals = load_accept(Path(args.accept))
    batches = int(acc["seeds_batches"])
    gpu_path, identity_path, meta_path = map(Path, (args.gpu, args.identity_map, args.gpu_meta))
    identity, reverse = load_identity_map(identity_path)
    meta = validate_gpu_metadata(meta_path, gpu_path, identity_path, len(identity))
    topas_meta_path = Path(args.topas_meta)
    topas_meta = validate_topas_metadata(topas_meta_path, identity_path,
                                         meta["source_records_sha256"], intervals)
    gpu_planes = load_gpu(gpu_path, identity, intervals)
    depmap = {}
    for i, dep in enumerate([intervals[0][0]] + [b for _, b in intervals]):
        depmap[dep] = i
    topas_planes = {}
    topas_files = []
    topas_headers = []
    for dep, plane_idx in depmap.items():
        p = Path(args.topas_dir) / f"{args.topas_prefix}_{int(round(dep)):03d}.phsp"
        topas_files.append(p)
        header_path = p.with_suffix(".header")
        topas_headers.append(validate_topas_header(header_path))
        topas_planes[plane_idx] = load_topas(p, dep, reverse, identity)
    require(set(depmap.values()) == set(gpu_planes) == set(topas_planes),
            "required GPU/TOPAS planes are incomplete")
    require(meta.get("required_intervals_mm") == [[a, b] for a, b in intervals],
            "GPU metadata required intervals do not match acceptance.yaml")
    res = {"schema": "maigo.replay-gate.v2", "state": "BLOCKED",
           "inputs": {"gpu": str(gpu_path), "gpu_sha256": sha256(gpu_path),
                      "gpu_meta": str(meta_path), "gpu_meta_sha256": sha256(meta_path),
                      "topas_meta": str(topas_meta_path),
                      "topas_meta_sha256": sha256(topas_meta_path),
                      "identity_map": str(identity_path),
                      "identity_map_sha256": sha256(identity_path),
                      "topas_files": [str(p) for p in topas_files],
                      "topas_sha256": [sha256(p) for p in topas_files],
                      "topas_headers": topas_headers,
                      "topas_config_sha256": topas_meta["config_sha256"],
                      "acceptance": str(Path(args.accept)),
                      "acceptance_sha256": sha256(Path(args.accept)),
                      "source_records": len(identity),
                      "source_records_sha256": meta["source_records_sha256"],
                      "config_sha256": meta["config_sha256"]},
           "acceptance": acc, "intervals": [], "issues": []}
    res["before_first_required_plane"] = {
        "gpu_source_records_not_at_first_plane": len(identity) - len(gpu_planes[0]),
        "topas_source_records_not_at_first_plane": len(identity) - len(topas_planes[0]),
        "meaning": "not observed at the 40 mm plane; may have stopped or escaped earlier"
    }
    any_fail, any_block = False, False
    min_total = max(1000, batches * 100)
    for j, (lo, hi) in enumerate(intervals):
        p0, p1 = j, j + 1
        gp0, gp1 = gpu_planes[p0], gpu_planes[p1]
        tp0, tp1 = topas_planes[p0], topas_planes[p1]
        gs, ts = set(gp0) & set(gp1), set(tp0) & set(tp1)
        common = sorted(gs & ts)
        row = {"from_mm": lo, "to_mm": hi,
               "gpu_population_at_start": len(gp0), "topas_population_at_start": len(tp0),
               "gpu_population_at_end": len(gp1), "topas_population_at_end": len(tp1),
               "gpu_missing_after_start": len(set(gp0) - set(gp1)),
               "topas_missing_after_start": len(set(tp0) - set(tp1)),
               "gpu_survival": _survival(gp0, gp1, identity, batches),
               "topas_survival": _survival(tp0, tp1, identity, batches),
               "gpu_only_survivors": len(gs - ts), "topas_only_survivors": len(ts - gs),
               "common_survivors": len(common)}
        if min(len(gs), len(ts), len(common)) < min_total:
            row["status"] = "INCONCLUSIVE"
            row["issues"] = [f"survivor samples below {min_total}"]
            any_block = True
            res["intervals"].append(row)
            continue
        def pairs(left, right, ids):
            return [np.stack((left[i], right[i])) for i in ids]
        gpu_all = _moments(pairs(gp0, gp1, sorted(gs)), np.array(sorted(gs)), batches)
        topas_all = _moments(pairs(tp0, tp1, sorted(ts)), np.array(sorted(ts)), batches)
        gpu_pair = _moments(pairs(gp0, gp1, common), np.array(common), batches)
        topas_pair = _moments(pairs(tp0, tp1, common), np.array(common), batches)
        row["unconditional_survivor_metrics"] = {"gpu": gpu_all, "topas": topas_all}
        row["common_survivor_metrics"] = {"gpu": gpu_pair, "topas": topas_pair}
        gates = {
            "angular_second_moment": _ratio_gate(gpu_pair, topas_pair, "theta2", float(acc["angvar_rel"]), batches),
            "scatter_residual_second_moment": _ratio_gate(gpu_pair, topas_pair, "scatter_residual2_mm2", float(acc["disvar_rel"]), batches),
            "q999_space_angle": _ratio_gate(gpu_pair, topas_pair, "theta_q999_mrad", float(acc["q999_rel"]), batches),
            "tail_probability_gt100mrad": _absolute_gate(gpu_pair, topas_pair, "theta_gt100mrad", float(acc["tail_prob_abs"]), batches),
        }
        s0, s1 = row["gpu_survival"]["survival_fraction"], row["topas_survival"]["survival_fraction"]
        surv_ratio = s0 / s1 if s1 > 0 else None
        gb = row["gpu_survival"]["batch_fractions"]
        tb = row["topas_survival"]["batch_fractions"]
        if len(gb) == batches and len(tb) == batches and all(v > 0 for v in tb):
            batch_ratios = np.asarray(gb) / np.asarray(tb)
            surv_se = float(np.std(batch_ratios, ddof=1) / math.sqrt(batches))
            margin = abs(surv_ratio - 1.0) + 2.0 * surv_se
            gates["survival"] = {"status": "PASS" if margin < float(acc["survival_rel"]) else "FAIL",
                                 "ratio": surv_ratio, "se": surv_se,
                                 "gate": float(acc["survival_rel"]), "equivalence_margin": margin}
        else:
            gates["survival"] = {"status": "INCONCLUSIVE", "ratio": None, "se": None}
        row["gates"] = gates
        row["status"] = ("FAIL" if any(g["status"] == "FAIL" for g in gates.values()) else
                         "INCONCLUSIVE" if any(g["status"] == "INCONCLUSIVE" for g in gates.values()) else "PASS")
        any_fail |= row["status"] == "FAIL"
        any_block |= row["status"] == "INCONCLUSIVE"
        res["intervals"].append(row)
    require(len(res["intervals"]) == len(intervals), "internal required-interval accounting error")
    res["state"] = "FAIL" if any_fail else "INCONCLUSIVE" if any_block else "PASS"
    code = 1 if any_fail else 2 if any_block else 0
    return res, code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--gpu-meta", required=True)
    ap.add_argument("--identity-map", required=True)
    ap.add_argument("--topas-meta", required=True)
    ap.add_argument("--topas-dir", required=True)
    ap.add_argument("--topas-prefix", default="primary")
    ap.add_argument("--accept", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        result, code = run(args)
    except Exception as exc:
        result = {"schema": "maigo.replay-gate.v2", "state": "BLOCKED",
                  "issues": [str(exc)]}
        code = 2
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    print(f"{result['state']}: wrote {args.out}")
    for issue in result.get("issues", []):
        print("BLOCKED:", issue)
    if result["state"] == "FAIL":
        print("one or more acceptance thresholds failed")
    return code


if __name__ == "__main__":
    sys.exit(main())
