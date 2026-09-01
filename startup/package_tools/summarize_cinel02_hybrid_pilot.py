#!/usr/bin/env python3
"""Summarize CINEL02 hybrid pilot raw headers without decoding product payloads."""

import argparse
import csv
import re
from pathlib import Path

import cinel02


PATTERN = re.compile(r"z(\d+)a(\d+)_e(\d+)_h(\d+)_v3$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for run in sorted(args.root.iterdir()):
        match = PATTERN.match(run.name)
        if not match:
            continue
        z, a, energy, histories = map(int, match.groups())
        interactions = products = raw_bytes = workers = 0
        for path in run.glob("raw/*/worker_*.cinel02"):
            with path.open("rb") as stream:
                header = stream.read(cinel02.RAW_HEADER_SIZE)
            if len(header) != cinel02.RAW_HEADER_SIZE:
                raise ValueError(f"truncated raw header: {path}")
            values = cinel02.RAW_HEADER.unpack(header)
            if values[0] != cinel02.RAW_MAGIC or values[1] != cinel02.RAW_VERSION:
                raise ValueError(f"invalid raw header: {path}")
            interactions += int(values[5]); products += int(values[6])
            raw_bytes += path.stat().st_size; workers += 1
        completed = False
        log = run / "topas.log"
        if log.is_file():
            completed = "TOPAS run sequence complete" in log.read_text(errors="replace")
        rows.append({"projectile_z": z, "projectile_a": a, "source_energy_MeV_per_u": energy,
                     "histories": histories, "completed": int(completed), "raw_workers": workers,
                     "interactions": interactions, "products": products, "raw_bytes": raw_bytes,
                     "interactions_per_history": interactions / histories})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"summarized {len(rows)} pilot points; completed={sum(r['completed'] for r in rows)}; interactions={sum(r['interactions'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
