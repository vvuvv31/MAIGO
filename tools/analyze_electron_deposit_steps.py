#!/usr/bin/env python3
"""Audit TOPAS electron ancestry and energy-weighted birth-to-deposit offsets.

This diagnoses a kernel; it does not generate or replace any physics package.
All positions are world coordinates. Midpoints approximate deposition within
each recorded step; pre/post endpoint summaries expose that approximation.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_slab_energy(d, root, tracks, bounds, tolerance=1e-8, birth_match=None):
    """Restricted diagnostic: single traversal, homogeneous box in vacuum.

    This is NOT a general geometry boundary scorer. Contiguous track step IDs
    and actual terminal face coordinates are required; re-entry is rejected.
    ASCII callers use 0.0006 mm tolerance for six-significant-digit coordinates;
    native binary callers use 1e-8 mm. Neither may relax outward direction.
    birth_match optionally maps electron roots to (ke, dir, distance) birth-
    time matches, echoed per family (null when absent).
    """
    bounds = np.asarray(bounds).reshape(3, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid slab bounds")
    if np.any(d[:, 19] != 1):
        raise ValueError("Escape audit requires analog unit-weight tracks")
    if np.any(~np.isin(d[:, 4], [1000060120, 11, 22])):
        raise ValueError("Escape audit supports EM C12/electron/photon only")
    for label, col in (("birth KE", 9), ("pre-step KE", 17), ("post-step KE", 18)):
        values = d[:, col]
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid {label}: must be finite and non-negative")
    order = np.lexsort((d[:, 5], d[:, 2], d[:, 1], d[:, 0]))
    ordered = d[order]
    change = np.any(ordered[1:, :3] != ordered[:-1, :3], axis=1)
    if np.any((np.diff(ordered[:, 5]) != 1) & ~change):
        raise ValueError("Missing/duplicate steps or re-entry in track")
    starts = np.r_[0, np.flatnonzero(change)+1]
    ends = np.r_[starts[1:]-1, len(ordered)-1]
    family = {}
    incoming = outgoing = primary_outgoing = 0.0
    escaped_tracks = 0
    # Per-event bookkeeping so global closure cannot hide cancellations.
    per_event = {}

    def on_face(p):
        inside = np.all((p >= bounds[:, 0]-tolerance) & (p <= bounds[:, 1]+tolerance))
        return inside and np.any(np.abs(p[:, None]-bounds) <= tolerance)

    for start, end in zip(starts, ends):
        first, last = ordered[start], ordered[end]
        key = tuple(first[:3].astype(int))
        rk = root(key)
        ev_key = (int(first[0]), int(first[1]))
        slot = per_event.setdefault(ev_key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # slot: incident, deposited, outgoing, family birth, family deposited,
        # family escaped.
        segment_deposit = float(ordered[start:end+1, 16].sum())
        slot[1] += segment_deposit
        if rk is not None:
            if rk not in family:
                # Count each root once, not once per descendant track.
                slot[3] += float(tracks[rk][3])
            fam = family.setdefault(rk, [float(tracks[rk][3]), 0.0, 0.0])
            if first[5] != 1 or not np.allclose(first[10:13], tracks[key][2], rtol=0, atol=tolerance):
                raise ValueError("Secondary birth/first-step gap or outside-born track")
            fam[1] += segment_deposit
            slot[4] += segment_deposit
        else:
            if not on_face(first[10:13]):
                raise ValueError("Primary input must enter slab at its boundary")
            incoming += first[17]
            slot[0] += float(first[17])
        if last[18] < 0:
            raise ValueError("Negative terminal kinetic energy")
        if last[18] > 0:
            if not on_face(last[13:16]):
                raise ValueError("Positive-KE terminal inside slab: truncated or killed track")
            direction = last[13:16]-last[10:13]
            low = np.abs(last[13:16]-bounds[:, 0]) <= tolerance
            high = np.abs(last[13:16]-bounds[:, 1]) <= tolerance
            if not np.any((low & (direction < 0)) | (high & (direction > 0))):
                raise ValueError(f"Terminal boundary direction is not outward: track={key}, "
                                 f"pre={last[10:13]}, post={last[13:16]}, KE={last[18]}")
            outgoing += last[18]
            slot[2] += float(last[18])
            escaped_tracks += 1
            if rk is None:
                primary_outgoing += last[18]
            else:
                family[rk][2] += last[18]
                slot[5] += float(last[18])
    if not family:
        raise ValueError("No electron families for escape audit")
    energy = np.asarray(list(family.values()))
    birth, deposited, escaped = energy.sum(axis=0)
    total = d[:, 16].sum()
    if incoming <= 0 or birth <= 0:
        raise ValueError("No incident or electron birth energy")
    global_residual = (incoming-total-outgoing)/incoming
    family_residual = (birth-deposited-escaped)/birth
    if abs(global_residual) > 1e-3 or abs(family_residual) > 1e-3:
        raise ValueError(f"Escape closure failed: total={global_residual}, electron={family_residual}")
    # Per-event residuals: the same closure must hold event by event so that
    # opposite-sign errors cannot cancel in the global sum.
    per_event_residuals = []
    for ev_key in sorted(per_event):
        inc, dep, out, birth_ev, dep_ev, esc_ev = per_event[ev_key]
        if inc <= 0:
            raise ValueError(f"Event {ev_key} has no incident energy")
        event_global = (inc-dep-out)/inc
        slot_family = None
        if birth_ev > 0:
            slot_family = (birth_ev-dep_ev-esc_ev)/birth_ev
            if abs(slot_family) > 1e-3:
                raise ValueError(
                    f"Escape closure failed in event {ev_key}: electron={slot_family}")
        if abs(event_global) > 1e-3:
            raise ValueError(
                f"Escape closure failed in event {ev_key}: total={event_global}")
        per_event_residuals.append({
            "run": int(ev_key[0]), "event": int(ev_key[1]),
            "incident_MeV": float(inc), "deposited_MeV": float(dep),
            "outgoing_MeV": float(out),
            "electron_birth_MeV": float(birth_ev),
            "electron_family_deposited_MeV": float(dep_ev),
            "electron_family_escaped_MeV": float(esc_ev),
            "global_relative_residual": float(event_global),
            "electron_relative_residual": float(slot_family) if slot_family is not None else None,
        })
    # Per-family residuals and worst-family identity (birth/deposit/escape).
    family_residual_list = []
    worst_key = None
    worst_abs = -1.0
    for rk, (birth_rk, dep_rk, esc_rk) in family.items():
        residual = (birth_rk-dep_rk-esc_rk)/birth_rk if birth_rk > 0 else None
        absolute = abs(birth_rk-dep_rk-esc_rk)
        if (residual is not None and abs(residual) > 1e-3) or (residual is None and absolute > 0):
            raise ValueError(f"Escape closure failed in family {rk}: residual={residual}, absolute={absolute}")
        if absolute > worst_abs:
            worst_abs = absolute
            worst_key = rk
        matched = (birth_match or {}).get(rk, (None, None, None))
        family_residual_list.append({
            "run": int(rk[0]), "event": int(rk[1]), "track": int(rk[2]),
            "birth_MeV": float(birth_rk),
            "deposited_MeV": float(dep_rk),
            "escaped_MeV": float(esc_rk),
            "relative_residual": float(residual) if residual is not None else None,
            "absolute_residual_MeV": float(absolute),
            "matched_parent_ke_MeV": float(matched[0]) if matched[0] is not None else None,
            "match_distance_mm": float(matched[2]) if matched[2] is not None else None,
        })
    if worst_key is None:
        raise ValueError("No electron families for escape audit")
    family_residual_list.sort(key=lambda item: item["absolute_residual_MeV"],
                              reverse=True)
    return {
        "scope": "EM C12 slab, vacuum exterior, no re-entry; terminal KE only",
        "coordinate_tolerance_mm": tolerance,
        "incident_MeV": float(incoming), "outgoing_MeV": float(outgoing),
        "primary_outgoing_MeV": float(primary_outgoing),
        "electron_root_count": len(family), "escaped_tracks": escaped_tracks,
        "electron_birth_MeV": float(birth),
        "electron_family_deposited_MeV": float(deposited),
        "electron_family_escaped_MeV": float(escaped),
        "global_relative_residual": float(global_residual),
        "electron_relative_residual": float(family_residual),
        "maximum_family_absolute_residual_MeV": float(np.max(np.abs(energy[:, 0]-energy[:, 1]-energy[:, 2]))),
        "per_event_residuals": per_event_residuals,
        "maximum_event_global_absolute_residual": float(
            max(abs(item["global_relative_residual"]) for item in per_event_residuals)),
        "per_family_residuals": family_residual_list,
        "worst_family": {
            "run": int(worst_key[0]), "event": int(worst_key[1]),
            "track": int(worst_key[2]),
            "absolute_residual_MeV": float(worst_abs),
        },
    }


def bind_generating_steps(d, root, tracks):
    """Verify v3 exact parent-step IDs; no nearest-neighbour fallback."""
    if np.any(d[:, 29] != np.floor(d[:, 29])):
        raise ValueError("Nonintegral parent step ID")
    valid = d[:, 22] == 1
    if np.any(d[valid, 29] < 1) or np.any(d[valid, 30] < 0):
        raise ValueError("Invalid generating-step/post-KE record")
    if np.any(np.abs(np.linalg.norm(d[valid, 31:34], axis=1)-1) > 1e-6):
        raise ValueError("Invalid generating-step post direction")
    if np.any(d[~valid, 29:31] != -1):
        raise ValueError("Invalid v3 parent sentinel")
    first = {}
    parent_steps = {}
    for row in d:
        key = tuple(row[:3].astype(int))
        first.setdefault(key, row)
        if row[3] == 0 and row[4] == 1000060120:
            step_key = (*key, int(row[5]))
            if step_key in parent_steps:
                raise ValueError("Duplicate generating parent step")
            parent_steps[step_key] = row
    result = {}
    for key in tracks:
        rk = root(key)
        if rk is None or rk in result:
            continue
        row = first[rk]
        if row[22] != 1:
            raise ValueError(f"Missing exact generating-step binding for {rk}")
        parent_key = (rk[0], rk[1], int(row[3]), int(row[29]))
        if parent_key not in parent_steps:
            raise ValueError("Generating parent step missing from ntuple")
        parent = parent_steps[parent_key]
        if not np.allclose([row[23], row[30]], parent[17:19], rtol=1e-10, atol=1e-8):
            raise ValueError("Generating parent step KE mismatch")
        vec = parent[13:16]-parent[10:13]
        lensq = float(np.dot(vec, vec))
        t = np.clip(np.dot(row[6:9]-parent[10:13], vec)/lensq, 0, 1) if lensq > 0 else 0
        dist = float(np.linalg.norm(row[6:9]-parent[10:13]-t*vec))
        result[rk] = (float(row[23]), row[24:27].copy(), dist)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=Path, required=True)
    p.add_argument("--dose", type=Path, required=True)
    p.add_argument("--voxel-volume-mm3", type=float, required=True)
    p.add_argument("--histories", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--format", choices=["ascii", "topas-binary-le"], default="ascii")
    p.add_argument("--slab-bounds-mm", type=float, nargs=6,
                   help="xmin xmax ymin ymax zmin zmax: vacuum exterior, no re-entry")
    a = p.parse_args()
    if a.histories <= 0 or not np.isfinite(a.voxel_volume_mm3) or a.voxel_volume_mm3 <= 0:
        raise ValueError("Positive histories and finite voxel volume required")
    if a.steps.stat().st_size > 512 * 1024**2:
        raise ValueError("Ntuple exceeds 512 MiB diagnostic input limit; use a smaller run")
    header_path = a.steps.with_suffix(".header")
    header = header_path.read_text()
    v1_columns = ["run", "event", "track", "parent", "pdg", "step",
                  "birth_x_mm", "birth_y_mm", "birth_z_mm", "birth_ke_MeV",
                  "pre_x_mm", "pre_y_mm", "pre_z_mm", "post_x_mm", "post_y_mm",
                  "post_z_mm", "edep_MeV", "pre_ke_MeV", "post_ke_MeV",
                  "weight", "density_g_cm3"]
    v2_extra = ["schema_version", "parent_valid", "parent_ke_MeV",
                "parent_dir_x", "parent_dir_y", "parent_dir_z",
                "creator_process_id", "birth_density_g_cm3"]
    v2_columns = v1_columns + v2_extra
    v3_columns = v2_columns + ["parent_step_id", "parent_post_ke_MeV",
                               "parent_post_dir_x", "parent_post_dir_y", "parent_post_dir_z"]
    binary = a.format == "topas-binary-le"
    pattern = r"^\s*([if]\d+):\s*(\S+)\s*$" if binary else r"^\s*(\d+):\s*(\S+)\s*$"
    actual = re.findall(pattern, header, re.MULTILINE)

    # v2 int columns: 0-5 (identity) + 21 schema_version, 22 parent_valid,
    # 27 creator_process_id; the rest are f8.
    def expected_for_typed(columns):
        if len(columns) == 21:
            int_positions = set(range(6))
        elif len(columns) in (29, 34):
            int_positions = set(range(6)) | {21, 22, 27}
            if len(columns) == 34:
                int_positions.add(29)
        else:
            int_positions = set()
        return [(("i4" if i in int_positions else "f8"), name)
                if binary else (str(i+1), name)
                for i, name in enumerate(columns)]

    schema_version = None
    if actual == expected_for_typed(v1_columns):
        columns = v1_columns
        schema_version = 1
    elif actual == expected_for_typed(v2_columns):
        columns = v2_columns
        schema_version = 2
    elif actual == expected_for_typed(v3_columns):
        columns = v3_columns
        schema_version = 3
    else:
        raise ValueError(
            "Unknown ntuple schema: header matches neither v1 (21 columns) nor "
            "v2 (29 columns) or v3 (34 columns) CarbonElectronDepositNtuple; refusing to parse")
    original = re.search(r"Number of Original Histories: (\d+)", header)
    entries = re.search(r"Number of Scored Entries: (\d+)", header)
    if original is None or int(original[1]) != a.histories or entries is None:
        raise ValueError("Header history count/completion mismatch")
    if binary:
        int_positions = set(range(6)) if schema_version == 1 else set(range(6)) | {21, 22, 27}
        if schema_version == 3:
            int_positions.add(29)
        dtype = np.dtype([(name, "<i4" if i in int_positions else "<f8")
                          for i, name in enumerate(columns)])
        nbytes = re.search(r"Number of Bytes per Particle: (\d+)", header)
        if sys.byteorder != "little" or nbytes is None or int(nbytes[1]) != dtype.itemsize:
            raise ValueError("Unsupported binary endian/record size")
        if a.steps.stat().st_size != int(entries[1])*dtype.itemsize:
            raise ValueError("Truncated ntuple: binary byte count mismatch")
        raw = np.memmap(a.steps, dtype=dtype, mode="r")
        d = np.column_stack([raw[name] for name in columns])
        del raw
    else:
        d = np.loadtxt(a.steps, ndmin=2)
    if len(d) != int(entries[1]):
        raise ValueError("Truncated ntuple: header entry count mismatch")
    if d.shape[1] not in (21, 29, 34) or not np.isfinite(d).all():
        raise ValueError("Expected 21 (v1), 29 (v2) or 34 (v3) finite columns "
                         "from CarbonElectronDepositNtuple")
    if d.shape[1] != len(columns):
        raise ValueError("Header/row column count mismatch: unknown schema")
    if np.any(d[:, 16] < 0) or np.any(d[:, 19:21] <= 0):
        raise ValueError("Invalid deposited energy, weight or material density")
    if np.any(~np.isfinite(d[:, 9])) or np.any(d[:, 9] < 0):
        raise ValueError("Invalid birth KE: must be finite and non-negative")
    if np.any(~np.isfinite(d[:, 17])) or np.any(d[:, 17] < 0):
        raise ValueError("Invalid pre-step KE: must be finite and non-negative")
    if np.any(~np.isfinite(d[:, 18])) or np.any(d[:, 18] < 0):
        raise ValueError("Invalid post-step KE: must be finite and non-negative")
    if not np.array_equal(d[:, :6], np.floor(d[:, :6])):
        raise ValueError("Nonintegral track identity columns")
    if len(set(map(tuple, d[:, :2].astype(int)))) != a.histories:
        raise ValueError("Incomplete event coverage")
    # Explicitly reconstruct ancestry: secondary electrons and photons must
    # remain referenced to the original delta electron, not their own births.
    # Per-track consistency: run/event/track/parent/PDG/birth position/birth KE
    # must be identical across all rows of the same track. For v2 the
    # parent-conditioning fields and creator/birth-density fields join the
    # per-track signature (the scorer replays the birth-time snapshot on every
    # step of the track).
    tracks = {}
    for row in d:
        key = tuple(row[:3].astype(int))
        signature = (int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                     int(row[4]), float(row[6]), float(row[7]),
                     float(row[8]), float(row[9]))
        if schema_version >= 2:
            signature = signature + (
                int(row[21]), int(row[22]), float(row[23]), float(row[24]),
                float(row[25]), float(row[26]), int(row[27]), float(row[28]))
        if schema_version == 3:
            signature += tuple(row[29:34])
        if key not in tracks:
            tracks[key] = (int(row[3]), int(row[4]), row[6:9].copy(), row[9],
                           signature)
        else:
            prev = tracks[key][4]
            if prev != signature:
                raise ValueError(
                    f"Inconsistent per-track identity/birth fields for track {key}: "
                    f"run/event/track/parent/PDG/birth position/birth KE must agree")
    if schema_version >= 2:
        if np.any(d[:, 21] != schema_version):
            raise ValueError(f"v{schema_version} record with schema_version != {schema_version}: unknown schema")
        if set(np.unique(d[:, 22].astype(int))) - {0, 1}:
            raise ValueError("v2 parent_valid must be 0 or 1")
        if np.any(~np.isin(d[:, 27].astype(int), list(range(10)))):
            raise ValueError("v2 creator_process_id out of legend range 0-9")
        if np.any(d[:, 28] <= 0):
            raise ValueError("v2 birth density must be positive")
        valid = d[:, 22] == 1
        if np.any(d[valid, 23] < 0):
            raise ValueError("v2 parent KE must be non-negative when valid")
        norms = np.linalg.norm(d[valid][:, 24:27], axis=1)
        if np.any(np.abs(norms - 1) > 1e-6):
            raise ValueError("v2 parent direction must be unit when valid")
        if np.any(d[~valid, 23] != -1.0):
            raise ValueError("v2 invalid rows must carry the -1 parent-KE sentinel")
    # Per-event primary count: this diagnostic defines exactly one primary
    # (parent == 0) track per (run, event).
    primaries_per_event = {}
    for key, info in tracks.items():
        if info[0] == 0:
            ev = (key[0], key[1])
            primaries_per_event[ev] = primaries_per_event.get(ev, 0) + 1
    observed_events = set(map(tuple, d[:, :2].astype(int)))
    for ev in observed_events:
        nprim = primaries_per_event.get((int(ev[0]), int(ev[1])), 0)
        if nprim != 1:
            raise ValueError(
                f"Event {ev} has {nprim} primary tracks; expected exactly 1")
    # Per-track duplicate step numbers are rejected explicitly (the slab audit
    # also requires contiguous step IDs; see audit_slab_energy).
    seen_steps = {}
    for row in d:
        key = tuple(row[:3].astype(int))
        step = int(row[5])
        bucket = seen_steps.setdefault(key, set())
        if step in bucket:
            raise ValueError(f"Duplicate step {step} in track {key}")
        bucket.add(step)
    roots = {}
    def root(key):
        if key in roots:
            return roots[key]
        chain = []
        cur = key
        while cur not in roots:
            if cur not in tracks:
                raise ValueError(f"Missing ancestor {cur}")
            if cur in chain:
                raise ValueError("Cyclic track ancestry")
            chain.append(cur)
            parent, pdg, birth, birth_ke = tracks[cur][:4]
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

    def match_birth_parents():
        """Bind each electron root to its creation-time C12 state (v2 only).

        The scorer-cached parent_* columns hold the event's LAST C12 step
        (exit proxy: identical for every electron in the event), NOT the
        creation-time state — Geant4 tracks the primary to completion before
        popping secondaries. True birth conditioning is recovered offline:
        the electron vertex must lie on its parent C12 step segment, so the
        nearest same-event C12 segment gives the creation-time KE (pre/post
        interpolation) and direction. Returns {root: (ke, dir, distance)}.
        """
        matched = {}
        if schema_version == 3:
            return bind_generating_steps(d, root, tracks)
        if schema_version != 2:
            return matched
        c12_mask = (d[:, 4] == 1000060120) & (d[:, 3] == 0)
        c12_idx = np.flatnonzero(c12_mask)
        # Electron roots via per-TRACK ancestry (never per-row: millions of
        # rows would make this pass Python-slow on large shards).
        root_keys = []
        for key in tracks:
            rk = root(key)
            if rk is not None and rk not in root_keys:
                root_keys.append(rk)
        by_event = {}
        for idx in c12_idx:
            by_event.setdefault((int(d[idx, 0]), int(d[idx, 1])), []).append(idx)
        for rk in root_keys:
            ev = (rk[0], rk[1])
            birth = tracks[rk][2]
            seg_idx = by_event.get(ev)
            if not seg_idx:
                matched[rk] = (None, None, None)
                continue
            seg = d[seg_idx]
            pre, post = seg[:, 10:13], seg[:, 13:16]
            vec = post - pre
            lensq = np.einsum("ij,ij->i", vec, vec)
            lensq = np.where(lensq < 1e-24, 1e-24, lensq)
            t = np.clip(np.einsum("ij,ij->i", birth[None, :] - pre, vec) / lensq, 0.0, 1.0)
            closest = pre + t[:, None] * vec
            dist = np.linalg.norm(birth[None, :] - closest, axis=1)
            best = int(np.argmin(dist))
            ke = float(seg[best, 17] + t[best] * (seg[best, 18] - seg[best, 17]))
            direction = vec[best] / np.sqrt(lensq[best])
            matched[rk] = (ke, direction.copy(), float(dist[best]))
        return matched

    birth_match = match_birth_parents()
    offsets = []
    weights = []
    # v2 parent-conditioned projection inputs, aligned with offsets/weights.
    # The parent direction/KE come from the offline birth-time match
    # (birth_match), NOT the scorer-cached exit proxy. Creator ids are still
    # read from the scorer record. Rows whose root has no birth match are
    # counted as missing-parent and excluded from projection (never zero).
    parent_dirs = []
    parent_kes = []
    parent_valid_flags = []
    creator_ids = []
    scorer_snapshot = {}
    if schema_version >= 2:
        for row in d:
            key = tuple(row[:3].astype(int))
            if key not in scorer_snapshot:
                scorer_snapshot[key] = (row[24:27].copy(), float(row[23]),
                                        int(row[22]), int(row[27]))
    direct = 0.0
    depositing_roots = set()
    for row in d:
        w = row[16] * row[19]
        if w == 0:
            continue
        root_key = root(tuple(row[:3].astype(int)))
        if root_key is None:
            direct += w
        else:
            depositing_roots.add(root_key)
            b = tracks[root_key][2]
            offsets.append(np.concatenate((row[10:13]-b, row[13:16]-b)))
            weights.append(w)
            if schema_version >= 2:
                ke, direction, dist = birth_match.get(root_key, (None, None, None))
                matched_ok = ke is not None and dist is not None and dist <= 1.0
                parent_dirs.append(direction if matched_ok else np.zeros(3))
                parent_kes.append(ke if matched_ok else -1.0)
                parent_valid_flags.append(1 if matched_ok else 0)
                creator_ids.append(scorer_snapshot[root_key][3])
    if not weights:
        raise ValueError("No deposited delta-family energy")
    off = np.asarray(offsets)
    w = np.asarray(weights)
    def quantiles(x):
        order = np.argsort(x)
        cdf = np.cumsum(w[order]) / w.sum()
        return np.interp([0.1, 0.5, 0.9, 0.99], cdf, x[order]).tolist()
    pre = off[:, 2]
    post = off[:, 5]
    mid = (pre+post)/2
    mid_xyz = (off[:, :3] + off[:, 3:])/2
    radial = np.linalg.norm(mid_xyz[:, :2], axis=1)
    radial_edges = np.array([0, .1, .5, 1, 2, 5, 10, 20, 50, 100, 300])
    longitudinal_edges = np.array([-500, -100, -20, -5, -1, -.5, 0, .5, 1, 5, 20, 100, 500])
    joint, _, _ = np.histogram2d(radial, mid, bins=(radial_edges, longitudinal_edges), weights=w)
    if not np.isclose(joint.sum(), w.sum(), rtol=1e-10):
        raise ValueError("Joint histogram overflow; widen diagnostic bins explicitly")
    # v2 parent-conditioned block: project each deposited midpoint onto the
    # recorded parent direction of its electron root. Rows whose root has
    # parent_valid == 0 are counted as missing-parent and excluded from the
    # projected quantiles (never zero-filled).
    parent_conditioning = None
    if schema_version >= 2:
        pdir = np.asarray(parent_dirs)
        pke = np.asarray(parent_kes)
        pvalid = np.asarray(parent_valid_flags)
        cids = np.asarray(creator_ids)
        missing = int(np.sum(pvalid == 0))
        # Distinct DEPOSITING electron roots, and how many lack a birth-time
        # parent match (match distance <= 1 mm required). Reuses the root set
        # collected in the offsets pass (no extra per-row ancestry walk).
        roots_seen = {}
        for rk in depositing_roots:
            ke, direction, dist = birth_match.get(rk, (None, None, None))
            roots_seen[rk] = 1 if (ke is not None and dist is not None
                                   and dist <= 1.0) else 0
        missing_roots = int(sum(1 for flag in roots_seen.values() if flag == 0))
        matched_distances = [birth_match[rk][2] for rk in roots_seen
                             if birth_match.get(rk, (None, None, None))[2] is not None]
        matched_kes = [birth_match[rk][0] for rk in roots_seen
                       if birth_match.get(rk, (None, None, None))[0] is not None]
        valid_rows = pvalid == 1
        if np.sum(valid_rows) == 0:
            projected = None
        else:
            mid_valid_xyz = mid_xyz[valid_rows]
            dir_valid = pdir[valid_rows]
            longitudinal = np.einsum("ij,ij->i", mid_valid_xyz, dir_valid)
            perp = mid_valid_xyz - longitudinal[:, None] * dir_valid
            radial_proj = np.linalg.norm(perp, axis=1)
            w_valid = w[valid_rows]
            order = np.argsort(longitudinal)
            cdf = np.cumsum(w_valid[order]) / w_valid.sum()
            projected = {
                "longitudinal_quantile_probabilities": [0.1, 0.5, 0.9, 0.99],
                "projected_longitudinal_mm": np.interp(
                    [0.1, 0.5, 0.9, 0.99], cdf, longitudinal[order]).tolist(),
                "projected_radial_median_mm": float(
                    np.interp(0.5, np.cumsum(
                        w_valid[np.argsort(radial_proj)]) / w_valid.sum(),
                        radial_proj[np.argsort(radial_proj)])),
                "valid_deposited_MeV": float(w_valid.sum()),
            }
        creator_values, creator_counts = np.unique(cids, return_counts=True)
        parent_conditioning = {
            "definition": ("longitudinal = (mid - birth) . parent_dir; "
                           "radial = |(mid - birth) - longitudinal*parent_dir|; "
                           "parent_dir/KE are the OFFLINE birth-time match "
                           "(nearest same-event C12 segment to the electron "
                           "vertex, <= 1 mm), never assumed +z and never the "
                           "scorer-cached exit proxy; scorer parent_* columns "
                           "hold the event's last C12 step (exit state) because "
                           "Geant4 completes the primary before popping "
                           "secondaries"),
            "deposited_rows": int(len(w)),
            "missing_parent_rows": missing,
            "electron_roots_with_deposit": len(roots_seen),
            "missing_parent_roots": missing_roots,
            "missing_parent_root_rate": float(missing_roots / len(roots_seen)) if roots_seen else None,
            "match_distance_mm_quantiles": (
                np.interp([0.1, 0.5, 0.9, 0.99],
                          np.arange(1, len(matched_distances) + 1) / len(matched_distances),
                          np.sort(np.asarray(matched_distances))).tolist()
                if matched_distances else None),
            "matched_parent_ke_quantiles_MeV": (
                np.interp([0.1, 0.5, 0.9, 0.99],
                          np.arange(1, len(matched_kes) + 1) / len(matched_kes),
                          np.sort(np.asarray(matched_kes))).tolist()
                if matched_kes else None),
            "parent_ke_valid_MeV": [float(v) for v in pke[valid_rows].tolist()] if np.sum(valid_rows) else [],
            "creator_process_id_counts": {str(int(k)): int(v) for k, v in zip(creator_values, creator_counts)},
            "projected": projected,
            "projected_status": ("computed" if projected is not None else
                                 "not computed: no valid-parent deposited energy"),
        }
    dose = np.loadtxt(a.dose, delimiter=",", comments="#", ndmin=2)
    if dose.shape[1] < 4 or not np.isfinite(dose).all() or np.any(dose[:, 3] < 0):
        raise ValueError("Invalid 3D dose data")
    rho = np.unique(d[:, 20])
    if len(rho) != 1:
        raise ValueError("Dose-energy closure requires homogeneous material")
    # rho[g/cm3] * V[mm3] * 1e-6 = voxel mass in kg.
    dose_energy = dose[:, 3].sum()*rho[0]*a.voxel_volume_mm3*1e-6*6.241509074e12
    total = float(np.sum(d[:, 16]*d[:, 19]))
    if abs(dose_energy/total - 1) > 1e-3:
        raise ValueError(f"3D dose/step energy closure failed: {dose_energy/total}")
    result = {
        "histories_requested": a.histories,
        "input_format": a.format,
        "parent_conditioning_runtime_eligible": False,
        "parent_conditioning_limitation": ("v3 exact generating-step association; pre/post are not process-internal sampling states; response validation pending" if schema_version == 3 else
            "v2 parent fields are end-of-track proxies; nearest-segment recovery is diagnostic"),
        "record_schema_version": schema_version,
        "material_density_g_cm3": float(rho[0]),
        "analyzer_sha256": file_sha256(Path(__file__)),
        "events_observed": len(set(map(tuple, d[:, :2].astype(int)))),
        "rows": len(d), "tracks": len(tracks),
        "total_deposit_MeV": total, "primary_local_MeV": direct,
        "delta_family_deposit_MeV": float(w.sum()),
        "delta_fraction_of_deposit": float(w.sum()/total),
        "dose3d_integrated_MeV": float(dose_energy),
        "dose3d_over_steps": float(dose_energy/total),
        "longitudinal_quantile_probabilities": [0.1, 0.5, 0.9, 0.99],
        "longitudinal_pre_mm": quantiles(pre),
        "longitudinal_mid_mm": quantiles(mid),
        "longitudinal_post_mm": quantiles(post),
        "forward_midpoint_energy_fraction": float(w[mid>0].sum()/w.sum()),
        "beyond_0p5mm_forward_fraction": float(w[mid>0.5].sum()/w.sum()),
        "joint_radial_edges_mm": radial_edges.tolist(),
        "joint_longitudinal_edges_mm": longitudinal_edges.tolist(),
        "joint_deposited_MeV": joint.tolist(),
        "parent_conditioning": parent_conditioning,
        "parent_conditioning_status": (
            ("computed: exact generating-step pre-state projection" if schema_version == 3 else
             "computed: approximate v2 parent projection") if schema_version >= 2 else
            "not computed: v1 record has no parent state (null, not zero)"),
        "steps_sha256": file_sha256(a.steps),
        "raw_steps": str(a.steps.resolve()),
        "raw_header": str(header_path.resolve()),
        "raw_dose": str(a.dose.resolve()),
        "header_sha256": file_sha256(header_path),
        "dose_sha256": file_sha256(a.dose),
        "scope": "Homogeneous diagnostic; not a compiled transport kernel",
        "limitations": [
            "Displacements are world-z offsets for a beam aligned along +z.",
            "Only deposited energy inside this finite slab is summarized; escaped energy is not a kernel sample.",
            "Primary energy lost below the production cut stays in the primary-local category.",
            "Step endpoints bracket an approximate deposit location; this is not microscopic electron tracking.",
            "Many correlated step rows do not replace independent primary histories or uncertainty estimation.",
        ],
    }
    if a.slab_bounds_mm is not None:
        result["finite_slab_energy_audit"] = audit_slab_energy(
            d, root, tracks, a.slab_bounds_mm, tolerance=1e-8 if binary else 0.0006,
            birth_match=birth_match if schema_version >= 2 else None)
        result["finite_slab_energy_audit_status"] = "computed"
    else:
        result["finite_slab_energy_audit"] = None
        result["finite_slab_energy_audit_status"] = (
            "not computed: --slab-bounds-mm not provided; no escape closure claimed")
    if schema_version == 3:
        result["parent_conditioning"]["definition"] = (
            "Exact generating parent step ID and recorded pre-step direction/KE; "
            "post-step state retained; no nearest-segment inference or process-internal KE claim")
        result["parent_step_binding_verified"] = True
    payload = json.dumps(result, indent=2, allow_nan=False)
    a.output.write_text(payload+"\n")
    print(payload)


if __name__ == "__main__":
    main()
