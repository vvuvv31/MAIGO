#!/usr/bin/env python3
"""Set voxels outside a float32 MHD mask to zero."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read(path: Path) -> tuple[dict[str, str], np.ndarray]:
    fields = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    shape = tuple(int(value) for value in fields["DimSize"].split())
    raw = np.fromfile(path.parent / fields["ElementDataFile"], dtype="<f4")
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} voxels, got {raw.size}")
    return fields, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    fields, values = read(args.input)
    mask_fields, mask = read(args.mask)
    if fields["DimSize"] != mask_fields["DimSize"]:
        raise ValueError(
            f"shape mismatch: input {fields['DimSize']} vs mask {mask_fields['DimSize']}"
        )
    masked = np.where(mask > 0.5, values, 0.0).astype("<f4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw")
    masked.tofile(raw_path)
    fields["ElementDataFile"] = raw_path.name
    fields["AppliedMask"] = str(args.mask)
    args.output.write_text(
        "\n".join(f"{key} = {value}" for key, value in fields.items()) + "\n",
        encoding="ascii",
    )
    print(
        f"{args.output}: retained {np.count_nonzero(mask > 0.5)} mask voxels; "
        f"sum {float(values.sum(dtype=np.float64)):.9g} -> "
        f"{float(masked.sum(dtype=np.float64)):.9g}"
    )


if __name__ == "__main__":
    main()
