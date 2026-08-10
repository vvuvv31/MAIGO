#!/usr/bin/env python3
"""Copy a CCTG grid while replacing only its low-edge origin metadata."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


HEADER = struct.Struct("<IIIIIffffff")
MAGIC = 0x47544343


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", nargs=3, type=float, required=True)
    args = parser.parse_args()

    payload = bytearray(args.input.read_bytes())
    if len(payload) < HEADER.size:
        raise SystemExit(f"Truncated CCTG header: {args.input}")
    fields = list(HEADER.unpack_from(payload))
    if fields[0] != MAGIC:
        raise SystemExit(f"Invalid CCTG magic: {args.input}")
    old = tuple(float(v) for v in fields[5:8])
    fields[5:8] = args.origin
    HEADER.pack_into(payload, 0, *fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {args.output}: origin {old} -> {tuple(args.origin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
