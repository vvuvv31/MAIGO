#!/usr/bin/env python3
"""See whether later same-vertex cascade products can fill primary B/Be/Li.

Reads the existing 1M cascade interaction/product CSVs (no new TOPAS run).
Primary packages currently keep only track-1 C-12 event_interaction_id==0.
This reports extra Z=3/4/5 born at the same vertex in later interactions of
the same event — prompt residual de-excitation that the extract dropped.
"""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


VERTEX_MM = 1.0e-3


def open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--energy-min-mevu", type=float, default=200.0)
    parser.add_argument("--energy-max-mevu", type=float, default=204.0)
    args = parser.parse_args()

    # event_key -> list of interaction dicts
    events = defaultdict(list)
    with open_csv(args.interactions) as stream:
        for row in csv.DictReader(stream):
            events[(int(row["run_id"]), int(row["thread_id"]), int(row["event_id"]))].append(
                {
                    "iid": int(row["interaction_id"]),
                    "seq": int(row["event_interaction_id"]),
                    "track": int(row["interaction_track_id"]),
                    "z": int(row["projectile_Z"]),
                    "a": int(row["projectile_A"]),
                    "energy": float(row["incident_energy_MeV_per_u"]),
                    "depth": float(row["depth_mm"]),
                    "process": row["process"],
                    "process_type": int(row["process_type"]),
                }
            )

    primaries = []
    follow_iids = set()
    primary_iids = set()
    energy_primaries = []
    process_counter = defaultdict(int)
    for members in events.values():
        first = None
        for item in members:
            if item["z"] == 6 and item["a"] == 12 and item["track"] == 1 and item["seq"] == 0:
                first = item
                break
        if first is None:
            continue
        primaries.append(first)
        primary_iids.add(first["iid"])
        in_energy = args.energy_min_mevu <= first["energy"] < args.energy_max_mevu
        if in_energy:
            energy_primaries.append(first)
        for item in members:
            if item["iid"] == first["iid"]:
                continue
            if abs(item["depth"] - first["depth"]) > VERTEX_MM:
                continue
            follow_iids.add(item["iid"])
            process_counter[item["process"]] += 1
            item["_primary_iid"] = first["iid"]
            item["_in_energy"] = in_energy

    # Map follow iid -> primary iid / energy flag
    follow_meta = {}
    for members in events.values():
        first = None
        for item in members:
            if item["z"] == 6 and item["a"] == 12 and item["track"] == 1 and item["seq"] == 0:
                first = item
                break
        if first is None:
            continue
        in_energy = args.energy_min_mevu <= first["energy"] < args.energy_max_mevu
        for item in members:
            if item["iid"] == first["iid"]:
                continue
            if abs(item["depth"] - first["depth"]) > VERTEX_MM:
                continue
            follow_meta[item["iid"]] = {
                "primary_iid": first["iid"],
                "in_energy": in_energy,
                "process": item["process"],
                "proj_z": item["z"],
                "proj_a": item["a"],
            }

    def bucket():
        return {
            "n": 0,
            "ke": 0.0,
            "n_fast": 0,  # KE > 200 MeV
            "ke_fast": 0.0,
        }

    id0 = {z: bucket() for z in (3, 4, 5, 6)}
    follow = {z: bucket() for z in (3, 4, 5, 6)}
    id0_e = {z: bucket() for z in (3, 4, 5, 6)}
    follow_e = {z: bucket() for z in (3, 4, 5, 6)}
    follow_by_process = defaultdict(
        lambda: {z: bucket() for z in (3, 4, 5, 6)}
    )
    id0_has = {z: 0 for z in (3, 4, 5)}
    follow_adds_when_missing = {z: 0 for z in (3, 4, 5)}
    primary_has_from_id0 = defaultdict(set)
    primary_has_from_follow = defaultdict(set)
    energy_primary_iids = {p["iid"] for p in energy_primaries}
    id0_e = {z: bucket() for z in (3, 4, 5, 6)}
    follow_e = {z: bucket() for z in (3, 4, 5, 6)}

    wanted = primary_iids | follow_iids
    with open_csv(args.products) as stream:
        for row in csv.DictReader(stream):
            iid = int(row["interaction_id"])
            if iid not in wanted:
                continue
            z = int(row["Z"])
            if z not in (3, 4, 5, 6):
                continue
            ke = float(row["kinetic_energy_MeV"])
            if iid in primary_iids:
                id0[z]["n"] += 1
                id0[z]["ke"] += ke
                if ke > 200.0:
                    id0[z]["n_fast"] += 1
                    id0[z]["ke_fast"] += ke
                if iid in energy_primary_iids:
                    id0_e[z]["n"] += 1
                    id0_e[z]["ke"] += ke
                    if ke > 200.0:
                        id0_e[z]["n_fast"] += 1
                        id0_e[z]["ke_fast"] += ke
                if z in (3, 4, 5):
                    primary_has_from_id0[iid].add(z)
            else:
                meta = follow_meta[iid]
                follow[z]["n"] += 1
                follow[z]["ke"] += ke
                if ke > 200.0:
                    follow[z]["n_fast"] += 1
                    follow[z]["ke_fast"] += ke
                follow_by_process[meta["process"]][z]["n"] += 1
                follow_by_process[meta["process"]][z]["ke"] += ke
                if ke > 200.0:
                    follow_by_process[meta["process"]][z]["n_fast"] += 1
                    follow_by_process[meta["process"]][z]["ke_fast"] += ke
                if meta["in_energy"]:
                    follow_e[z]["n"] += 1
                    follow_e[z]["ke"] += ke
                    if ke > 200.0:
                        follow_e[z]["n_fast"] += 1
                        follow_e[z]["ke_fast"] += ke
                if z in (3, 4, 5):
                    primary_has_from_follow[meta["primary_iid"]].add(z)

    for iid, zs in primary_has_from_id0.items():
        for z in (3, 4, 5):
            if z in zs:
                id0_has[z] += 1
    for iid, zs in primary_has_from_follow.items():
        have = primary_has_from_id0.get(iid, set())
        for z in (3, 4, 5):
            if z in zs and z not in have:
                follow_adds_when_missing[z] += 1

    n_pri = len(primaries)
    n_e = len(energy_primaries)
    print(f"primary C12 track1 id0 reactions: {n_pri}")
    print(f"in [{args.energy_min_mevu},{args.energy_max_mevu}) MeV/u: {n_e}")
    print(f"same-vertex later interactions: {len(follow_iids)}")
    print("later-interaction processes (same vertex):")
    for name, count in sorted(process_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}")

    def report(title: str, table: dict, denom: int) -> None:
        print(f"\n{title}  (per primary reaction, denom={denom})")
        print(f"{'Z':>4} {'mult':>8} {'<KE>':>8} {'fast_mult':>10} {'fast_<KE>':>9}")
        if denom == 0:
            return
        for z in (3, 4, 5, 6):
            n = table[z]["n"]
            ke = table[z]["ke"]
            nf = table[z]["n_fast"]
            kef = table[z]["ke_fast"]
            print(
                f"{z:4d} {n/denom:8.4f} {ke/n if n else 0:8.1f} "
                f"{nf/denom:10.4f} {kef/nf if nf else 0:9.1f}"
            )

    report("id0 products (all energies)", id0, n_pri)
    report("same-vertex later products (all energies)", follow, n_pri)
    report(f"id0 products [{args.energy_min_mevu},{args.energy_max_mevu})", id0_e, n_e)
    report(
        f"same-vertex later [{args.energy_min_mevu},{args.energy_max_mevu})",
        follow_e,
        n_e,
    )

    print("\nreactions that already have Z in id0 / later adds Z when id0 lacked it:")
    for z in (3, 4, 5):
        print(
            f"  Z={z}: id0_has={id0_has[z]} ({id0_has[z]/n_pri:.3f})  "
            f"later_adds={follow_adds_when_missing[z]} "
            f"({follow_adds_when_missing[z]/n_pri:.3f})"
        )

    print("\nlater products by process (all energies):")
    for name, table in sorted(
        follow_by_process.items(), key=lambda kv: -sum(v["n"] for v in kv[1].values())
    ):
        parts = []
        for z in (3, 4, 5, 6):
            if table[z]["n"]:
                parts.append(
                    f"Z{z} n={table[z]['n']} fast={table[z]['n_fast']} "
                    f"<KE>={table[z]['ke']/table[z]['n']:.1f}"
                )
        print(f"  {name}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
