#!/usr/bin/env python3
"""Scale a little-endian MET_FLOAT MHD dose while preserving its geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError("--scale must be finite and positive")

    lines = args.input.read_text(encoding="ascii").splitlines()
    meta = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    if meta.get("ElementType") != "MET_FLOAT":
        raise ValueError("only MET_FLOAT input is supported")
    dims = [int(value) for value in meta["DimSize"].split()]
    raw_in = args.input.parent / meta["ElementDataFile"]
    data = np.fromfile(raw_in, dtype="<f4")
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(f"{raw_in}: expected {expected} floats, got {data.size}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_out = args.output.with_suffix(".raw")
    scaled = (data.astype(np.float64) * args.scale).astype("<f4")
    scaled.tofile(raw_out)

    output_lines = []
    for line in lines:
        if line.startswith("ElementDataFile"):
            output_lines.append(f"ElementDataFile = {raw_out.name}")
        else:
            output_lines.append(line)
    output_lines.extend(
        [
            f"DoseScaleFactor = {args.scale:.17g}",
            f"SourceDoseMHD = {args.input}",
        ]
    )
    args.output.write_text("\n".join(output_lines) + "\n", encoding="ascii")

    report = {
        "input_mhd": str(args.input),
        "input_raw": str(raw_in),
        "output_mhd": str(args.output),
        "output_raw": str(raw_out),
        "scale": args.scale,
        "input_max_Gy": float(np.max(data)),
        "output_max_Gy": float(np.max(scaled)),
        "input_sum_Gy": float(np.sum(data, dtype=np.float64)),
        "output_sum_Gy": float(np.sum(scaled, dtype=np.float64)),
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
