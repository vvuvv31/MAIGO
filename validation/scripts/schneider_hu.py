#!/usr/bin/env python3
"""Parse TOPAS HUtoMaterialSchneider.txt and convert HU → density / material class.

Density formula (TOPAS / Schneider):
  Density = (Offset + Factor * (FactorOffset + HU)) * DensityCorrection[HU - HU_min]

Material class for GPU tables (air/lung/water/bone) is collapsed from Schneider
material sections by HU range — density still follows the full Schneider curve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SchneiderTable:
    hu_min: int
    density_correction: np.ndarray  # length = n_hu, g/cm3 factor
    density_section_bounds: np.ndarray  # len = n_sec + 1
    density_offset: np.ndarray  # len = n_sec
    density_factor: np.ndarray
    density_factor_offset: np.ndarray
    material_section_bounds: np.ndarray  # len = n_mat + 1
    # GPU class per Schneider material section index
    material_class: np.ndarray  # uint8, len = n_mat

    @property
    def hu_max_inclusive(self) -> int:
        return self.hu_min + int(self.density_correction.size) - 1


_VECTOR_RE = re.compile(
    r"^(?P<type>[a-z]{1,2}v?):(?P<path>\S+)\s*=\s*(?P<body>.+)$",
    re.IGNORECASE,
)


def _parse_vector_values(body: str) -> list[float] | None:
    # body: "N v1 v2 ..." possibly ending with a unit token
    # Strip quoted strings (not numeric vectors)
    if '"' in body:
        return None
    parts = body.replace("\t", " ").split()
    cleaned: list[str] = []
    for token in parts:
        low = token.lower()
        if low in {"g/cm3", "mm", "cm", "ev", "deg"}:
            continue
        if low.endswith("g/cm3"):
            token = token[: -len("g/cm3")]
            if not token:
                continue
        cleaned.append(token)
    if not cleaned:
        return None
    try:
        count = int(float(cleaned[0]))
        vals = [float(x) for x in cleaned[1 : 1 + count]]
        if len(vals) == count:
            return vals
        # fall through if count doesn't match
    except ValueError:
        pass
    try:
        return [float(x) for x in cleaned]
    except ValueError:
        return None


def load_schneider_table(path: Path) -> SchneiderTable:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    params: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _VECTOR_RE.match(line)
        if not match:
            continue
        key = match.group("path")
        short = key.split("/")[-1] if "/" in key else key
        body = match.group("body").strip()
        ptype = match.group("type").lower()
        # Only numeric vectors for density/material section tables
        if ptype.startswith("s"):
            continue
        parsed = _parse_vector_values(body)
        if parsed is not None:
            params[short] = parsed

    if "DensityCorrection" not in params:
        raise ValueError(f"DensityCorrection missing in {path}")
    dens_corr = np.asarray(params["DensityCorrection"], dtype=np.float64)
    hu_min = -1000  # TOPAS default MinImagingValue
    if "MinImagingValue" in params and params["MinImagingValue"]:
        hu_min = int(params["MinImagingValue"][0])

    dens_bounds = np.asarray(
        params["SchneiderHounsfieldUnitSections"], dtype=np.int32
    )
    dens_offset = np.asarray(params["SchneiderDensityOffset"], dtype=np.float64)
    dens_factor = np.asarray(params["SchneiderDensityFactor"], dtype=np.float64)
    dens_foff = np.asarray(
        params["SchneiderDensityFactorOffset"], dtype=np.float64
    )
    n_sec = len(dens_bounds) - 1
    if not (
        len(dens_offset) == n_sec
        and len(dens_factor) == n_sec
        and len(dens_foff) == n_sec
    ):
        raise ValueError("Schneider density section vector lengths mismatch")
    if dens_bounds[-1] - dens_bounds[0] != dens_corr.size:
        # TOPAS requires range == number of DensityCorrection values
        raise ValueError(
            f"DensityCorrection length {dens_corr.size} != HU range "
            f"{dens_bounds[-1] - dens_bounds[0]}"
        )

    mat_bounds = np.asarray(params["SchneiderHUToMaterialSections"], dtype=np.int32)
    n_mat = len(mat_bounds) - 1
    # Collapse Schneider tissue sections → GPU 4-class tables used in transport.
    # Section 0: air; early negative HU: lung; soft tissue → water; Ca-rich → bone.
    mat_class = np.zeros(n_mat, dtype=np.uint8)
    for i in range(n_mat):
        lo = int(mat_bounds[i])
        hi = int(mat_bounds[i + 1])  # exclusive-ish upper of section
        mid_hu = 0.5 * (lo + hi)
        if mid_hu < -950:
            mat_class[i] = 0  # air
        elif mid_hu < -120:
            mat_class[i] = 1  # lung
        elif mid_hu < 80:
            mat_class[i] = 2  # soft / water-equivalent
        else:
            mat_class[i] = 3  # bone

    return SchneiderTable(
        hu_min=hu_min,
        density_correction=dens_corr,
        density_section_bounds=dens_bounds,
        density_offset=dens_offset,
        density_factor=dens_factor,
        density_factor_offset=dens_foff,
        material_section_bounds=mat_bounds,
        material_class=mat_class,
    )


def build_density_lut(table: SchneiderTable) -> np.ndarray:
    """Precompute density [g/cm3] for each integer HU in the table range."""
    n = table.density_correction.size
    hu = np.arange(table.hu_min, table.hu_min + n, dtype=np.float64)
    dens = np.empty(n, dtype=np.float64)
    bounds = table.density_section_bounds
    for s in range(len(bounds) - 1):
        lo = int(bounds[s])
        hi = int(bounds[s + 1])
        # section covers HU in [lo, hi)
        mask = (hu >= lo) & (hu < hi)
        if not np.any(mask):
            continue
        off = table.density_offset[s]
        fac = table.density_factor[s]
        foff = table.density_factor_offset[s]
        idx = (hu[mask] - table.hu_min).astype(np.int64)
        dens[mask] = (off + fac * (foff + hu[mask])) * table.density_correction[idx]
    dens = np.clip(dens, 1.0e-6, 10.0)
    return dens.astype(np.float32)


def build_material_lut(table: SchneiderTable) -> np.ndarray:
    n = table.density_correction.size
    hu = np.arange(table.hu_min, table.hu_min + n, dtype=np.int32)
    mat = np.full(n, 2, dtype=np.uint8)
    bounds = table.material_section_bounds
    for s in range(len(bounds) - 1):
        lo = int(bounds[s])
        hi = int(bounds[s + 1])
        mask = (hu >= lo) & (hu < hi)
        mat[mask] = table.material_class[s]
    # last HU bin
    if hu[-1] >= bounds[-2]:
        mat[-1] = table.material_class[-1]
    return mat


def hu_to_density_material(
    hu: np.ndarray, table: SchneiderTable
) -> tuple[np.ndarray, np.ndarray]:
    dens_lut = build_density_lut(table)
    mat_lut = build_material_lut(table)
    hu_i = np.rint(hu.astype(np.float64)).astype(np.int64)
    hu_i = np.clip(hu_i, table.hu_min, table.hu_min + dens_lut.size - 1)
    idx = (hu_i - table.hu_min).astype(np.int64)
    return dens_lut[idx], mat_lut[idx]
