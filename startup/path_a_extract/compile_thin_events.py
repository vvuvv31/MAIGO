#!/usr/bin/env python3
"""Pack thin-target CarbonReaction n-tuples into FELB v1 event libraries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import struct

MAXF = 8
NKEEP = 8192


def parse_phsp(path: Path) -> dict[int, list]:
    events: dict[int, list] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        kind = parts[0]
        if kind not in {"reaction", "secondary"}:
            continue
        event = int(parts[2])
        z = int(parts[7])
        a = int(parts[8])
        ke = float(parts[10])
        dx, dy, dz = map(float, parts[15:18])
        events[event].append((kind, z, a, ke, dx, dy, dz))
    return events


def pack(target: str, z: int, a: int, energy: float, src: Path, out_dir: Path) -> None:
    events = parse_phsp(src)
    keys = sorted(events)
    kept = []
    stride = max(1, len(keys) // NKEEP)
    for key in keys[::stride]:
        frags = []
        nke = 0.0
        for kind, zz, aa, ke, dx, dy, dz in events[key]:
            if kind == "reaction":
                continue
            nrm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
            if zz == 0 and aa == 1:
                nke += ke
            elif zz >= 1 and aa >= 1:
                frags.append((zz, aa, ke, dx, dy, dz))
        if not frags:
            continue
        if len(frags) > MAXF:
            frags.sort(key=lambda item: item[2], reverse=True)
            nke += sum(item[2] for item in frags[MAXF:])
            frags = frags[:MAXF]
        kept.append((frags, nke))
        if len(kept) >= NKEEP:
            break
    payload = bytearray()
    payload += b"FELB"
    payload += struct.pack("<IIIfHH", 1, len(kept), MAXF, float(energy), z, a)
    for frags, nke in kept:
        payload += struct.pack("<Bf", len(frags), nke)
        for i in range(MAXF):
            if i < len(frags):
                zz, aa, ke, dx, dy, dz = frags[i]
                payload += struct.pack("<bbffff", zz, aa, ke, dx, dy, dz)
            else:
                payload += struct.pack("<bbffff", 0, 0, 0.0, 0.0, 0.0, 1.0)
    out = out_dir / f"c12_{target}_{int(energy)}MeVu_events.bin"
    out.write_bytes(payload)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "target": target,
                "Z": z,
                "A": a,
                "energy_MeVu": energy,
                "events": len(kept),
                "source_events": len(keys),
                "bytes": len(payload),
            },
            indent=2,
        )
        + "\n"
    )
    print(out, "events", len(kept), "from", len(keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy", type=float, default=95.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e = int(args.energy)
    for target, z, a in (("H1", 1, 1), ("C12", 6, 12), ("O16", 8, 16)):
        src = (
            args.extract_root
            / target
            / f"e{e}"
            / f"c12_{target}_{e}mevu_thin_reactions.phsp"
        )
        pack(target, z, a, args.energy, src, args.output_dir)


if __name__ == "__main__":
    main()
