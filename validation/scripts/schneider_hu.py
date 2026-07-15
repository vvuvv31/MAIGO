#!/usr/bin/env python3
"""Parse TOPAS HUtoMaterialSchneider.txt → density, section id, mass-SP factors.

Density (TOPAS / Schneider):
  Density = (Offset + Factor * (FactorOffset + HU)) * DensityCorrection[HU - HU_min]

Mass stopping-power factor (energy-independent Bragg Z/A proxy vs water):
  mass_sp_factor[section] = (Z/A)_section / (Z/A)_water
so device SP ≈ SP_water_table(E) * mass_sp_factor[section] * density.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Z/A for Schneider element names (atomic number / atomic mass).
_ELEMENT_Z_OVER_A: dict[str, float] = {
    "Hydrogen": 1.0 / 1.00794,
    "Carbon": 6.0 / 12.0107,
    "Nitrogen": 7.0 / 14.0067,
    "Oxygen": 8.0 / 15.999,
    "Magnesium": 12.0 / 24.305,
    "Phosphorus": 15.0 / 30.9738,
    "Sulfur": 16.0 / 32.065,
    "Chlorine": 17.0 / 35.453,
    "Argon": 18.0 / 39.948,
    "Calcium": 20.0 / 40.078,
    "Sodium": 11.0 / 22.9898,
    "Potassium": 19.0 / 39.0983,
    "Titanium": 22.0 / 47.867,
}

# Approximate mean excitation energies I [eV] (ICRU-like).
_ELEMENT_I_EV: dict[str, float] = {
    "Hydrogen": 19.2,
    "Carbon": 81.0,
    "Nitrogen": 82.0,
    "Oxygen": 95.0,
    "Magnesium": 156.0,
    "Phosphorus": 173.0,
    "Sulfur": 180.0,
    "Chlorine": 174.0,
    "Argon": 188.0,
    "Calcium": 191.0,
    "Sodium": 149.0,
    "Potassium": 190.0,
    "Titanium": 233.0,
}

# Liquid water reference (ICRU-style mass fractions).
_WATER_Z_OVER_A = 0.111894 * _ELEMENT_Z_OVER_A["Hydrogen"] + 0.888106 * _ELEMENT_Z_OVER_A[
    "Oxygen"
]


@dataclass
class SchneiderTable:
    hu_min: int
    density_correction: np.ndarray
    density_section_bounds: np.ndarray
    density_offset: np.ndarray
    density_factor: np.ndarray
    density_factor_offset: np.ndarray
    material_section_bounds: np.ndarray
    # Optional 4-class collapse (legacy / diagnostics)
    material_class: np.ndarray
    # Schneider section index 0..n_mat-1
    n_material_sections: int = 0
    elements: list[str] = field(default_factory=list)
    # shape (n_mat, n_elements) mass fractions
    material_weights: np.ndarray | None = None
    # High-E mass SP factor (Z/A ratio vs water), length n_mat
    mass_sp_factor: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float32))
    # Bragg mean I [eV] per section
    mass_sp_I_eV: np.ndarray = field(default_factory=lambda: np.full(1, 75.0, dtype=np.float32))

    @property
    def hu_max_inclusive(self) -> int:
        return self.hu_min + int(self.density_correction.size) - 1


_VECTOR_RE = re.compile(
    r"^(?P<type>[a-z]{1,2}v?):(?P<path>\S+)\s*=\s*(?P<body>.+)$",
    re.IGNORECASE,
)


def _parse_vector_values(body: str) -> list[float] | None:
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
    except ValueError:
        pass
    try:
        return [float(x) for x in cleaned]
    except ValueError:
        return None


def _parse_string_vector(body: str) -> list[str]:
    tokens = re.findall(r'"([^"]*)"', body)
    if tokens:
        return tokens
    parts = body.split()
    if parts and parts[0].isdigit():
        return parts[1:]
    return parts


def mass_sp_factor_from_weights(weights: np.ndarray, elements: list[str]) -> float:
    """Bragg Z/A mass-SP factor relative to liquid water (unitless, high-E limit)."""
    za, _ = composition_za_and_I(weights, elements)
    return float(za / _WATER_Z_OVER_A)


def composition_za_and_I(
    weights: np.ndarray, elements: list[str]
) -> tuple[float, float]:
    """Return ((Z/A)_eff, I_mean_eV) from mass fractions (Bragg ln I)."""
    if weights.size != len(elements):
        raise ValueError("weight/element length mismatch")
    num_za = 0.0
    num_lnI = 0.0
    den = 0.0
    for w, name in zip(weights, elements):
        if w <= 0.0:
            continue
        key = name.strip()
        za = _ELEMENT_Z_OVER_A.get(key, _ELEMENT_Z_OVER_A["Oxygen"])
        Iev = _ELEMENT_I_EV.get(key, 95.0)
        ww = float(w)
        num_za += ww * za
        num_lnI += ww * za * float(np.log(Iev))
        den += ww * za
    if den <= 0.0:
        return _WATER_Z_OVER_A, 75.0
    za_eff = num_za / max(sum(float(w) for w in weights if w > 0), 1e-30)
    # Better: mass-weighted Z/A
    wsum = sum(float(w) for w in weights if w > 0)
    za_eff = num_za / wsum if wsum > 0 else _WATER_Z_OVER_A
    I_mean = float(np.exp(num_lnI / den))
    return za_eff, I_mean


def load_schneider_table(path: Path) -> SchneiderTable:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    params: dict[str, list[float]] = {}
    strings: dict[str, list[str]] = {}
    weight_rows: dict[int, list[float]] = {}
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
        if ptype.startswith("s"):
            strings[short] = _parse_string_vector(body)
            continue
        parsed = _parse_vector_values(body)
        if parsed is None:
            continue
        m = re.match(r"SchneiderMaterialsWeight(\d+)$", short)
        if m:
            weight_rows[int(m.group(1))] = parsed
        else:
            params[short] = parsed

    if "DensityCorrection" not in params:
        raise ValueError(f"DensityCorrection missing in {path}")
    dens_corr = np.asarray(params["DensityCorrection"], dtype=np.float64)
    hu_min = -1000
    if "MinImagingValue" in params and params["MinImagingValue"]:
        hu_min = int(params["MinImagingValue"][0])

    dens_bounds = np.asarray(params["SchneiderHounsfieldUnitSections"], dtype=np.int32)
    dens_offset = np.asarray(params["SchneiderDensityOffset"], dtype=np.float64)
    dens_factor = np.asarray(params["SchneiderDensityFactor"], dtype=np.float64)
    dens_foff = np.asarray(params["SchneiderDensityFactorOffset"], dtype=np.float64)
    n_sec = len(dens_bounds) - 1
    if not (
        len(dens_offset) == n_sec
        and len(dens_factor) == n_sec
        and len(dens_foff) == n_sec
    ):
        raise ValueError("Schneider density section vector lengths mismatch")
    if dens_bounds[-1] - dens_bounds[0] != dens_corr.size:
        raise ValueError(
            f"DensityCorrection length {dens_corr.size} != HU range "
            f"{dens_bounds[-1] - dens_bounds[0]}"
        )

    mat_bounds = np.asarray(params["SchneiderHUToMaterialSections"], dtype=np.int32)
    n_mat = len(mat_bounds) - 1
    mat_class = np.zeros(n_mat, dtype=np.uint8)
    for i in range(n_mat):
        lo = int(mat_bounds[i])
        hi = int(mat_bounds[i + 1])
        mid_hu = 0.5 * (lo + hi)
        if mid_hu < -950:
            mat_class[i] = 0
        elif mid_hu < -120:
            mat_class[i] = 1
        elif mid_hu < 80:
            mat_class[i] = 2
        else:
            mat_class[i] = 3

    elements = strings.get("SchneiderElements", [])
    weights = np.zeros((n_mat, max(len(elements), 1)), dtype=np.float64)
    mass_factors = np.ones(n_mat, dtype=np.float32)
    I_eV = np.full(n_mat, 75.0, dtype=np.float32)
    if elements and weight_rows:
        for i in range(n_mat):
            row = weight_rows.get(i + 1)
            if row is None:
                continue
            w = np.asarray(row, dtype=np.float64)
            if w.size < len(elements):
                w = np.pad(w, (0, len(elements) - w.size))
            w = w[: len(elements)]
            s = w.sum()
            if s > 0:
                w = w / s
            weights[i, : len(elements)] = w
            za, Imean = composition_za_and_I(w, elements)
            mass_factors[i] = float(za / _WATER_Z_OVER_A)
            I_eV[i] = float(Imean)
    mass_factors = np.clip(mass_factors, 0.5, 1.5).astype(np.float32)
    I_eV = np.clip(I_eV, 10.0, 500.0).astype(np.float32)

    return SchneiderTable(
        hu_min=hu_min,
        density_correction=dens_corr,
        density_section_bounds=dens_bounds,
        density_offset=dens_offset,
        density_factor=dens_factor,
        density_factor_offset=dens_foff,
        material_section_bounds=mat_bounds,
        material_class=mat_class,
        n_material_sections=n_mat,
        elements=list(elements),
        material_weights=weights,
        mass_sp_factor=mass_factors,
        mass_sp_I_eV=I_eV,
    )


def build_density_lut(table: SchneiderTable) -> np.ndarray:
    n = table.density_correction.size
    hu = np.arange(table.hu_min, table.hu_min + n, dtype=np.float64)
    dens = np.empty(n, dtype=np.float64)
    bounds = table.density_section_bounds
    for s in range(len(bounds) - 1):
        lo = int(bounds[s])
        hi = int(bounds[s + 1])
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


def build_section_lut(table: SchneiderTable) -> np.ndarray:
    """HU → Schneider material section index (0..n_mat-1)."""
    n = table.density_correction.size
    hu = np.arange(table.hu_min, table.hu_min + n, dtype=np.int32)
    sec = np.zeros(n, dtype=np.uint8)
    bounds = table.material_section_bounds
    for s in range(len(bounds) - 1):
        lo = int(bounds[s])
        hi = int(bounds[s + 1])
        mask = (hu >= lo) & (hu < hi)
        sec[mask] = np.uint8(min(s, 255))
    sec[-1] = np.uint8(min(len(bounds) - 2, 255))
    return sec


def build_material_lut(table: SchneiderTable) -> np.ndarray:
    """Legacy 4-class collapse."""
    n = table.density_correction.size
    hu = np.arange(table.hu_min, table.hu_min + n, dtype=np.int32)
    mat = np.full(n, 2, dtype=np.uint8)
    bounds = table.material_section_bounds
    for s in range(len(bounds) - 1):
        lo = int(bounds[s])
        hi = int(bounds[s + 1])
        mask = (hu >= lo) & (hu < hi)
        mat[mask] = table.material_class[s]
    if hu[-1] >= bounds[-2]:
        mat[-1] = table.material_class[-1]
    return mat


def hu_to_density_material(
    hu: np.ndarray, table: SchneiderTable, *, use_section_id: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    dens_lut = build_density_lut(table)
    mat_lut = build_section_lut(table) if use_section_id else build_material_lut(table)
    hu_i = np.rint(hu.astype(np.float64)).astype(np.int64)
    hu_i = np.clip(hu_i, table.hu_min, table.hu_min + dens_lut.size - 1)
    idx = (hu_i - table.hu_min).astype(np.int64)
    return dens_lut[idx], mat_lut[idx]
