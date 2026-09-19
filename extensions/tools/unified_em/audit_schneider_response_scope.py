#!/usr/bin/env python3
"""Validate material-section identity and corrected Schneider density.

Density-formula intervals are NOT material sections. This guard is required
before interpreting different HU runs as a same-composition scaling test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def schneider_identity(path, hu):
    if not isinstance(hu, int):
        raise ValueError("HU must be an integer")
    source = Path(path).read_text()
    def vector(name):
        matches = re.findall(r"^[a-z]+:Ge/Patient/" + re.escape(name) + r"\s*=\s*(.*?)\s*$",
                             source, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError("Missing/duplicate Schneider parameter: " + name)
        words = matches[0].split()
        n = int(words[0])
        if name == "DensityCorrection" and words[-1] == "g/cm3":
            words = words[:-1]
        values = np.asarray([float(x) for x in words[1:]])
        if len(values) != n or not np.isfinite(values).all():
            raise ValueError("Invalid Schneider vector: " + name)
        return values
    material = vector("SchneiderHUToMaterialSections")
    density = vector("SchneiderHounsfieldUnitSections")
    correction = vector("DensityCorrection")
    offset = vector("SchneiderDensityOffset")
    factor = vector("SchneiderDensityFactor")
    factor_offset = vector("SchneiderDensityFactorOffset")
    if len(material) != 26 or np.any(np.diff(material) <= 0) or np.any(np.diff(density) <= 0):
        raise ValueError("Invalid Schneider section edges")
    if not all(len(v) == len(density)-1 for v in (offset, factor, factor_offset)):
        raise ValueError("Invalid density formula sizes")
    if not material[0] <= hu < material[-1] or not density[0] <= hu < density[-1]:
        raise ValueError("HU outside Schneider domain")
    if len(correction) != int(density[-1]-density[0]):
        raise ValueError("Invalid DensityCorrection domain")
    section = int(np.searchsorted(material, hu, side="right")-1)
    segment = int(np.searchsorted(density, hu, side="right")-1)
    corr = float(correction[hu-int(density[0])])
    rho = float((offset[segment]+factor[segment]*(factor_offset[segment]+hu))*corr)
    if rho <= 0 or corr <= 0:
        raise ValueError("Invalid corrected density")
    return {"hu": hu, "material_section": section,
            "material_hu_bounds": material[section:section+2].astype(int).tolist(),
            "density_formula_segment": segment,
            "density_correction": corr, "density_g_cm3": rho,
            "schneider_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schneider-file", type=Path,
                   default=Path(__file__).resolve().parents[1]/"data/HUtoMaterialSchneider.txt")
    p.add_argument("--hu", type=int, nargs="+", required=True)
    p.add_argument("--required-section", type=int, default=0)
    a = p.parse_args()
    values = [schneider_identity(a.schneider_file, hu) for hu in a.hu]
    print(json.dumps(values, indent=2, allow_nan=False))
    if any(v["material_section"] != a.required_section for v in values):
        raise SystemExit("Material-section scope mismatch: density-formula intervals cannot substitute for material sections")


if __name__ == "__main__":
    main()
