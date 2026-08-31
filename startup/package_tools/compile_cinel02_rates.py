#!/usr/bin/env python3
"""Compile TOPAS exposure measurements into a strict CINEL02 rate table.

The runtime CSV contains target-specific macroscopic hazards in mm^-1.  The
input is an exposure campaign: collision counts are divided by the weighted
track length through the same target/material, including histories that did
not collide.  This deliberately keeps rate extraction separate from the
whole-event CINPKG03 compiler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RATE_HEADER = (
    "projectile_z",
    "projectile_a",
    "target_z",
    "target_a",
    "energy_MeV_per_u",
    "macroscopic_cross_section_per_mm",
)

DEFAULT_MINIMUM_EVENTS_PER_BIN = 100
DEFAULT_MINIMUM_EFFECTIVE_COLLISIONS = 0.0
DEFAULT_MAXIMUM_RELATIVE_STANDARD_ERROR: float | None = None
DEFAULT_MINIMUM_EXPOSURE_MM = 0.0
DEFAULT_MAXIMUM_ZERO_RATE_UPPER_PER_MM = math.inf


def _canonical_campaign_uuid(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("rate campaign provenance must declare a campaign UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"invalid rate campaign UUID: {value!r}") from error
    canonical = str(parsed)
    if value.lower() != canonical:
        raise ValueError("rate campaign UUID must use canonical hyphenated form")
    return canonical


def _value(row: dict[str, str], names: Iterable[str], *, required: bool = True) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    if required:
        raise ValueError(f"rate input is missing one of: {', '.join(names)}")
    return None


def _number(row: dict[str, str], names: Iterable[str], *, required: bool = True) -> float | None:
    value = _value(row, names, required=required)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric rate field {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite rate field {value!r}")
    return parsed


def _integer(row: dict[str, str], names: Iterable[str]) -> int:
    value = _number(row, names)
    assert value is not None
    if value < 0.0 or value != math.floor(value):
        raise ValueError(f"rate identity/count must be a non-negative integer: {value}")
    return int(value)


def _optional_number(row: dict[str, str], names: Iterable[str]) -> float | None:
    return _number(row, names, required=False)


def _optional_integer(row: dict[str, str], names: Iterable[str]) -> int | None:
    value = _optional_number(row, names)
    if value is None:
        return None
    if value < 0.0 or value != math.floor(value):
        raise ValueError(f"rate bin/history field must be a non-negative integer: {value}")
    return int(value)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        lines = (line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError(f"rate input has no header: {path}")
        return [
            {str(key).strip(): (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]


def _effective_count(sum_weight: float, sum_weight_squared: float) -> float:
    if sum_weight_squared <= 0.0:
        return 0.0
    return sum_weight * sum_weight / sum_weight_squared


def compile_rates(
    input_paths: list[Path],
    output_path: Path,
    metadata_path: Path,
    *,
    minimum_events_per_bin: int = DEFAULT_MINIMUM_EVENTS_PER_BIN,
    minimum_effective_collision_count: float = DEFAULT_MINIMUM_EFFECTIVE_COLLISIONS,
    maximum_relative_standard_error: float | None = DEFAULT_MAXIMUM_RELATIVE_STANDARD_ERROR,
    minimum_exposure_mm: float = DEFAULT_MINIMUM_EXPOSURE_MM,
    maximum_zero_rate_upper_per_mm: float = DEFAULT_MAXIMUM_ZERO_RATE_UPPER_PER_MM,
    maximum_energy_grid_gap_mevu: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one or more exposure CSV files and return the sidecar object."""

    if minimum_events_per_bin <= 0:
        raise ValueError("minimum_events_per_bin must be positive")
    if minimum_effective_collision_count < 0.0 or not math.isfinite(
        minimum_effective_collision_count
    ):
        raise ValueError("minimum_effective_collision_count must be finite and non-negative")
    if maximum_relative_standard_error is not None and (
        maximum_relative_standard_error <= 0.0
        or not math.isfinite(maximum_relative_standard_error)
    ):
        raise ValueError("maximum_relative_standard_error must be positive and finite")
    if minimum_exposure_mm < 0.0 or not math.isfinite(minimum_exposure_mm):
        raise ValueError("minimum_exposure_mm must be finite and non-negative")
    if maximum_zero_rate_upper_per_mm <= 0.0 or not (
        math.isfinite(maximum_zero_rate_upper_per_mm)
        or math.isinf(maximum_zero_rate_upper_per_mm)
    ):
        raise ValueError("maximum_zero_rate_upper_per_mm must be positive")
    if maximum_energy_grid_gap_mevu is not None and (
        maximum_energy_grid_gap_mevu <= 0.0
        or not math.isfinite(maximum_energy_grid_gap_mevu)
    ):
        raise ValueError("maximum_energy_grid_gap_mevu must be positive and finite")
    if not input_paths:
        raise ValueError("at least one exposure CSV is required")

    # key -> [raw count, sum(w), sum(w²), exposure, density, areal density,
    #         history count, censored count, energy low, energy high,
    #         history sum(w), history sum(w²), censored sum(w),
    #         censored sum(w²)]
    # The integer bin id is part of the key when supplied. This prevents two
    # textual representations of the same floating-point edge from creating
    # separate runtime samples.
    aggregate: dict[tuple[int, int, int, int, int, float], list[float]] = defaultdict(
        lambda: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            math.nan, math.nan, 0.0, 0.0, 0.0, 0.0,
        ]
    )
    bin_definitions: dict[tuple[int, int, int, int, int], tuple[float, float, float]] = {}
    # A runtime CSV cell is identified by projectile/target identity and its
    # energy centre.  Keep the optional source bin id as an audit key, but do
    # not permit two different source bins (or a mixed id/no-id representation)
    # to emit duplicate runtime samples at one energy.
    runtime_bin_ids: dict[tuple[int, int, int, int, float], int | None] = {}
    input_hashes: list[dict[str, str]] = []
    for path in input_paths:
        input_hashes.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        for row in _rows(path):
            projectile_z = _integer(row, ("projectile_z", "projectile_Z"))
            projectile_a = _integer(row, ("projectile_a", "projectile_A"))
            target_z = _integer(row, ("target_z", "target_Z"))
            target_a = _integer(row, ("target_a", "target_A"))
            energy = _number(
                row, ("energy_MeV_per_u", "energy_bin", "collision_energy_MeV_per_u")
            )
            exposure = _number(
                row,
                (
                    "weighted_track_length_mm",
                    "track_length_weighted_mm",
                    "exposure_length_mm",
                ),
            )
            count = _number(row, ("inelastic_count", "collision_count", "collisions"))
            sum_weight = _number(row, ("sum_weight", "collision_weight"), required=False)
            sum_weight_squared = _number(
                row, ("sum_weight_squared", "collision_weight_squared"), required=False
            )
            density = _number(
                row,
                ("target_number_density_per_mm3", "number_density_per_mm3"),
                required=False,
            )
            history_count = _optional_integer(
                row, ("history_count", "histories", "history_total")
            )
            censored_history_count = _optional_integer(
                row, ("censored_history_count", "censored_histories", "censored")
            )
            history_weight_sum = _optional_number(
                row, ("history_weight_sum", "history_weight", "history_sum_weight")
            )
            history_weight_squared_sum = _optional_number(
                row,
                (
                    "history_weight_squared_sum",
                    "history_weight_squared",
                    "history_sum_weight_squared",
                ),
            )
            censored_history_weight_sum = _optional_number(
                row,
                (
                    "censored_history_weight_sum",
                    "censored_history_weight",
                    "censored_sum_weight",
                ),
            )
            censored_history_weight_squared_sum = _optional_number(
                row,
                (
                    "censored_history_weight_squared_sum",
                    "censored_history_weight_squared",
                    "censored_sum_weight_squared",
                ),
            )
            energy_bin_id = _optional_integer(row, ("energy_bin_id", "bin_id"))
            energy_low = _optional_number(
                row, ("energy_low_MeV_per_u", "energy_low", "energy_bin_low")
            )
            energy_high = _optional_number(
                row, ("energy_high_MeV_per_u", "energy_high", "energy_bin_high")
            )
            areal_density = _optional_number(
                row,
                (
                    "target_areal_density_weighted_per_mm2",
                    "weighted_target_areal_density_per_mm2",
                    "target_areal_density_per_mm2",
                ),
            )
            assert energy is not None and exposure is not None and count is not None
            if (
                projectile_z <= 0
                or projectile_a < projectile_z
                or target_z <= 0
                or target_a < target_z
                or energy < 0.0
                or exposure <= 0.0
                or count < 0.0
            ):
                raise ValueError(f"rate exposure row is outside its physical domain: {row}")
            if (energy_low is None) != (energy_high is None):
                raise ValueError("energy_low and energy_high must be supplied together")
            if energy_low is not None and (
                energy_low < 0.0
                or energy_high is None
                or energy_high <= energy_low
                or energy < energy_low - 1.0e-9
                or energy > energy_high + 1.0e-9
            ):
                raise ValueError(f"rate energy bin edges do not contain the sample: {row}")
            if history_count is None:
                history_count = 0
            if censored_history_count is None:
                censored_history_count = 0
            if censored_history_count > history_count:
                raise ValueError(f"censored histories exceed history count: {row}")
            if history_weight_sum is None and history_weight_squared_sum is None:
                history_weight_sum = float(history_count)
                history_weight_squared_sum = float(history_count)
            elif history_weight_sum is None or history_weight_squared_sum is None:
                raise ValueError(
                    "weighted history rows must provide both history weight moments: "
                    f"{row}"
                )
            if censored_history_weight_sum is None and censored_history_weight_squared_sum is None:
                censored_history_weight_sum = float(censored_history_count)
                censored_history_weight_squared_sum = float(censored_history_count)
            elif (
                censored_history_weight_sum is None
                or censored_history_weight_squared_sum is None
            ):
                raise ValueError(
                    "weighted censored-history rows must provide both weight moments: "
                    f"{row}"
                )
            if (
                history_weight_sum < 0.0
                or history_weight_squared_sum < 0.0
                or censored_history_weight_sum < 0.0
                or censored_history_weight_squared_sum < 0.0
                or (
                    history_weight_sum > 0.0
                    and history_weight_squared_sum <= 0.0
                )
                or (
                    censored_history_weight_sum > 0.0
                    and censored_history_weight_squared_sum <= 0.0
                )
                or censored_history_weight_sum > history_weight_sum + 1.0e-12
            ):
                raise ValueError(f"rate history weight moments are invalid: {row}")
            if count != math.floor(count):
                raise ValueError(f"raw collision count must be an integer: {row}")
            if areal_density is not None and areal_density <= 0.0:
                raise ValueError(f"target areal density must be positive: {row}")
            weighted_collisions = count if sum_weight is None else sum_weight
            if sum_weight is None and sum_weight_squared is None:
                # With no explicit weight columns, each counted collision has
                # unit weight. Zero-collision exposure remains valid.
                weighted_collisions_squared = count
            elif sum_weight_squared is None:
                if abs(weighted_collisions - count) > max(1.0e-12, abs(count) * 1.0e-9):
                    raise ValueError(
                        "weighted rate rows must provide sum_weight_squared: "
                        f"{row}"
                    )
                weighted_collisions_squared = count
            else:
                weighted_collisions_squared = sum_weight_squared
            if (
                weighted_collisions < 0.0
                or weighted_collisions_squared < 0.0
                or (weighted_collisions > 0.0 and weighted_collisions_squared <= 0.0)
            ):
                raise ValueError(f"rate collision weights are invalid: {row}")
            if count > 0.0 and weighted_collisions <= 0.0:
                raise ValueError(f"positive collisions must have positive weight: {row}")
            if count <= 0.0 and weighted_collisions > 0.0:
                raise ValueError(f"positive collision weight requires a collision count: {row}")
            bin_token = -1 if energy_bin_id is None else energy_bin_id
            runtime_key = (projectile_z, projectile_a, target_z, target_a, energy)
            previous_bin_id = runtime_bin_ids.get(runtime_key)
            if runtime_key in runtime_bin_ids:
                if (previous_bin_id is None) != (energy_bin_id is None):
                    raise ValueError(
                        "an energy cell must consistently provide energy_bin_id: "
                        f"{row}"
                    )
                if previous_bin_id is not None and previous_bin_id != energy_bin_id:
                    raise ValueError(
                        "duplicate runtime energy cell has conflicting energy_bin_id: "
                        f"{row}"
                    )
            else:
                runtime_bin_ids[runtime_key] = energy_bin_id
            key = (projectile_z, projectile_a, target_z, target_a, bin_token, energy)
            if energy_bin_id is not None:
                definition_key = (projectile_z, projectile_a, target_z, target_a, energy_bin_id)
                definition = (
                    energy_low if energy_low is not None else energy,
                    energy_high if energy_high is not None else energy,
                    energy,
                )
                previous_definition = bin_definitions.get(definition_key)
                if previous_definition is not None and any(
                    abs(lhs - rhs) > 1.0e-9
                    for lhs, rhs in zip(previous_definition, definition)
                ):
                    raise ValueError(f"energy bin id has inconsistent edges/center: {row}")
                bin_definitions[definition_key] = definition
            slot = aggregate[key]
            slot[0] += count
            slot[1] += weighted_collisions
            slot[2] += weighted_collisions_squared
            slot[3] += exposure
            slot[6] += history_count
            slot[7] += censored_history_count
            slot[10] += history_weight_sum
            slot[11] += history_weight_squared_sum
            slot[12] += censored_history_weight_sum
            slot[13] += censored_history_weight_squared_sum
            if energy_low is not None:
                if math.isfinite(slot[8]) and abs(slot[8] - energy_low) > 1.0e-9:
                    raise ValueError(f"inconsistent energy_low for {key}")
                if math.isfinite(slot[9]) and abs(slot[9] - energy_high) > 1.0e-9:
                    raise ValueError(f"inconsistent energy_high for {key}")
                slot[8] = energy_low
                slot[9] = energy_high
            if density is not None:
                if density <= 0.0:
                    raise ValueError(f"target number density must be positive: {row}")
                slot[4] = density if slot[4] == 0.0 else slot[4]
                if abs(slot[4] - density) > max(1.0e-12, abs(density) * 1.0e-6):
                    raise ValueError(f"inconsistent target number density for {key}")
            derived_areal_density = exposure * density if density is not None else None
            if areal_density is not None:
                if derived_areal_density is not None and abs(
                    areal_density - derived_areal_density
                ) > max(1.0e-9, abs(derived_areal_density) * 1.0e-6):
                    raise ValueError(f"target areal density disagrees with exposure: {row}")
                slot[5] += areal_density
            elif derived_areal_density is not None:
                slot[5] += derived_areal_density

    if not aggregate:
        raise ValueError("rate exposure input contains no rows")

    output_rows: list[dict[str, Any]] = []
    sidecar_samples: list[dict[str, Any]] = []
    groups: dict[tuple[int, int, int, int], list[dict[str, float]]] = defaultdict(list)
    for key in sorted(aggregate):
        (
            projectile_z,
            projectile_a,
            target_z,
            target_a,
            energy_bin_id,
            energy,
        ) = key
        (
            count,
            sum_weight,
            sum_weight_squared,
            exposure,
            density,
            areal_density,
            history_count,
            censored_history_count,
            energy_low,
            energy_high,
            history_weight_sum,
            history_weight_squared_sum,
            censored_history_weight_sum,
            censored_history_weight_squared_sum,
        ) = aggregate[key]
        if exposure < minimum_exposure_mm:
            raise ValueError(
                f"rate exposure cell {key} has {exposure:g} mm < "
                f"minimum {minimum_exposure_mm:g} mm"
            )
        effective_count = _effective_count(sum_weight, sum_weight_squared)
        rate = sum_weight / exposure
        zero_rate = sum_weight <= 0.0
        if not zero_rate and count < minimum_events_per_bin:
            raise ValueError(
                f"sparse rate exposure cell {key}: {count:g} < {minimum_events_per_bin}"
            )
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"invalid macroscopic rate for {key}")
        # A zero-rate cell has no observed collision sample, so its RSE is
        # undefined; the one-sided Poisson upper limit below is the useful
        # qualification quantity for that cell.
        relative_error = 1.0 / math.sqrt(effective_count) if effective_count > 0.0 else None
        zero_rate_upper = -math.log(0.05) / exposure if zero_rate else 0.0
        if zero_rate:
            if zero_rate_upper > maximum_zero_rate_upper_per_mm:
                raise ValueError(
                    f"covered zero-rate cell {key} has 95% upper limit "
                    f"{zero_rate_upper:g} > {maximum_zero_rate_upper_per_mm:g}"
                )
        else:
            if effective_count < minimum_effective_collision_count:
                raise ValueError(
                    f"low effective collision count for {key}: "
                    f"{effective_count:g} < {minimum_effective_collision_count:g}"
                )
            if (
                maximum_relative_standard_error is not None
                and relative_error > maximum_relative_standard_error
            ):
                raise ValueError(
                    f"relative standard error for {key} is {relative_error:g} > "
                    f"{maximum_relative_standard_error:g}"
                )
        if zero_rate:
            interval = zero_rate_upper
        else:
            assert relative_error is not None
            interval = 1.96 * rate * relative_error
        ci_low = max(0.0, rate - interval)
        ci_high = rate + interval
        if math.isfinite(energy_low) and math.isfinite(energy_high):
            sample_low = energy_low
            sample_high = energy_high
        else:
            sample_low = energy
            sample_high = energy
        output_rows.append(
            {
                "projectile_z": projectile_z,
                "projectile_a": projectile_a,
                "target_z": target_z,
                "target_a": target_a,
                "energy_MeV_per_u": energy,
                "macroscopic_cross_section_per_mm": rate,
            }
        )
        sample = {
            **key_to_dict(key),
            "collision_count": count,
            "sum_weight": sum_weight,
            "sum_weight_squared": sum_weight_squared,
            "effective_collision_count": effective_count,
            "relative_standard_error": relative_error,
            "weighted_track_length_mm": exposure,
            "target_areal_density_weighted_per_mm2": areal_density,
            "history_count": history_count,
            "censored_history_count": censored_history_count,
            "history_weight_sum": history_weight_sum,
            "history_weight_squared_sum": history_weight_squared_sum,
            "effective_history_count": _effective_count(
                history_weight_sum, history_weight_squared_sum
            ),
            "censored_history_weight_sum": censored_history_weight_sum,
            "censored_history_weight_squared_sum": censored_history_weight_squared_sum,
            "censoring_fraction": (
                censored_history_count / history_count if history_count > 0.0 else None
            ),
            "weighted_censoring_fraction": (
                censored_history_weight_sum / history_weight_sum
                if history_weight_sum > 0.0
                else None
            ),
            "macroscopic_cross_section_per_mm": rate,
            "interval_95_percent_per_mm": [ci_low, ci_high],
            "ci95_low_per_mm": ci_low,
            "ci95_high_per_mm": ci_high,
            "coverage_status": "covered_zero_rate" if zero_rate else "covered",
            "zero_rate_upper_95_per_mm": zero_rate_upper,
        }
        if energy_bin_id >= 0:
            sample["energy_bin_id"] = energy_bin_id
        if math.isfinite(energy_low) and math.isfinite(energy_high):
            sample["energy_low_MeV_per_u"] = energy_low
            sample["energy_high_MeV_per_u"] = energy_high
        if density > 0.0:
            sample["target_number_density_per_mm3"] = density
        if areal_density > 0.0:
            sample["microscopic_cross_section_mm2"] = sum_weight / areal_density
        sidecar_samples.append(sample)
        groups[(projectile_z, projectile_a, target_z, target_a)].append(
            {"energy": energy, "low": sample_low, "high": sample_high}
        )

    group_metadata = []
    for group, samples in sorted(groups.items()):
        ordered = sorted(samples, key=lambda sample: (sample["low"], sample["energy"]))
        maximum_gap = 0.0
        for previous, current in zip(ordered, ordered[1:]):
            if current["low"] < previous["high"] - 1.0e-9:
                raise ValueError(
                    f"overlapping energy bins for {group}: "
                    f"{previous['low']:g}..{previous['high']:g} and "
                    f"{current['low']:g}..{current['high']:g}"
                )
            maximum_gap = max(maximum_gap, current["low"] - previous["high"])
        maximum_gap = max(0.0, maximum_gap)
        if maximum_energy_grid_gap_mevu is not None and maximum_gap > (
            maximum_energy_grid_gap_mevu + 1.0e-9
        ):
            raise ValueError(
                f"energy grid gap for {group} is {maximum_gap:g} > "
                f"{maximum_energy_grid_gap_mevu:g}"
            )
        energies = [sample["energy"] for sample in ordered]
        group_metadata.append(
            {
                **key_to_dict(group),
                "sample_count": len(ordered),
                "minimum_energy_MeV_per_u": min(energies),
                "maximum_energy_MeV_per_u": max(energies),
                "maximum_gap_MeV_per_u": maximum_gap,
                "energy_bin_edges_present": any(
                    sample["low"] != sample["energy"] or sample["high"] != sample["energy"]
                    for sample in ordered
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RATE_HEADER)
        writer.writeheader()
        writer.writerows(output_rows)

    campaign_uuid = ""
    if provenance is not None and provenance.get("campaign_uuid") not in (None, ""):
        campaign_uuid = _canonical_campaign_uuid(provenance["campaign_uuid"])
    positive_samples = sum(
        sample["coverage_status"] == "covered" for sample in sidecar_samples
    )
    covered_zero_rate_samples = sum(
        sample["coverage_status"] == "covered_zero_rate" for sample in sidecar_samples
    )
    sidecar = {
        "format": "CINEL02_RATE_V1",
        "campaign_uuid": campaign_uuid,
        "runtime_header": list(RATE_HEADER),
        "minimum_events_per_bin": minimum_events_per_bin,
        "qualification": {
            "minimum_raw_collision_count": minimum_events_per_bin,
            "minimum_effective_collision_count": minimum_effective_collision_count,
            "maximum_relative_standard_error": maximum_relative_standard_error,
            "minimum_exposure_mm": minimum_exposure_mm,
            "maximum_zero_rate_upper_per_mm": (
                maximum_zero_rate_upper_per_mm
                if math.isfinite(maximum_zero_rate_upper_per_mm)
                else None
            ),
            "maximum_energy_grid_gap_MeV_per_u": maximum_energy_grid_gap_mevu,
            "qualified": True,
            "covered_samples": positive_samples,
            "covered_zero_rate_samples": covered_zero_rate_samples,
        },
        "groups": group_metadata,
        "samples": sidecar_samples,
        "input_files": input_hashes,
        "provenance": provenance or {},
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def key_to_dict(
    key: tuple[int, int, int, int]
    | tuple[int, int, int, int, float]
    | tuple[int, int, int, int, int, float],
) -> dict[str, Any]:
    result = {
        "projectile_z": key[0],
        "projectile_a": key[1],
        "target_z": key[2],
        "target_a": key[3],
    }
    if len(key) == 5:
        result["energy_MeV_per_u"] = key[4]
    elif len(key) == 6:
        if key[4] >= 0:
            result["energy_bin_id"] = key[4]
        result["energy_MeV_per_u"] = key[5]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument(
        "--minimum-events-per-bin",
        type=int,
        default=DEFAULT_MINIMUM_EVENTS_PER_BIN,
    )
    parser.add_argument(
        "--minimum-effective-collision-count",
        type=float,
        default=DEFAULT_MINIMUM_EFFECTIVE_COLLISIONS,
    )
    parser.add_argument("--maximum-relative-standard-error", type=float)
    parser.add_argument(
        "--minimum-exposure-mm",
        type=float,
        default=DEFAULT_MINIMUM_EXPOSURE_MM,
    )
    parser.add_argument(
        "--maximum-zero-rate-upper-per-mm",
        type=float,
        default=DEFAULT_MAXIMUM_ZERO_RATE_UPPER_PER_MM,
    )
    parser.add_argument("--maximum-energy-grid-gap-mevu", type=float)
    parser.add_argument("--campaign-uuid", required=True)
    parser.add_argument("--topas-version", default="")
    parser.add_argument("--geant4-version", default="")
    parser.add_argument("--physics-list", default="")
    parser.add_argument("--hadronic-process", default="ionInelastic")
    parser.add_argument("--hadronic-model", default="INCLXX")
    parser.add_argument("--production-cuts", "--cuts", dest="production_cuts", default="")
    parser.add_argument("--step-limits", default="")
    parser.add_argument("--material", default="G4_WATER")
    parser.add_argument("--density-g-cm3", type=float, default=1.0)

    def isotope(value: str) -> tuple[int, int, str]:
        try:
            atomic_number, mass_number, name = value.split(":", 2)
            parsed = int(atomic_number), int(mass_number), name
        except ValueError as error:
            raise argparse.ArgumentTypeError("isotope must be Z:A:NAME") from error
        if parsed[0] <= 0 or parsed[1] < parsed[0] or not parsed[2]:
            raise argparse.ArgumentTypeError("isotope must be a positive Z:A:NAME")
        return parsed

    parser.add_argument("--isotope", action="append", type=isotope, default=[])
    parser.add_argument("--source", default="")
    parser.add_argument("--capture-extension-revision", default="")
    args = parser.parse_args()
    provenance = {
        "campaign_uuid": args.campaign_uuid,
        "topas_version": args.topas_version,
        "geant4_version": args.geant4_version,
        "physics_list": args.physics_list,
        "hadronic_process": args.hadronic_process,
        "hadronic_model": args.hadronic_model,
        "production_cuts": args.production_cuts,
        "cuts": args.production_cuts,
        "step_limits": args.step_limits,
        "material": {
            "name": args.material,
            "density_g_cm3": args.density_g_cm3,
            "isotopes": [
                {"z": atomic_number, "a": mass_number, "name": name}
                for atomic_number, mass_number, name in args.isotope
            ],
        },
        "source": args.source,
        "capture_extension_revision": args.capture_extension_revision,
    }
    try:
        result = compile_rates(
            args.input,
            args.output,
            args.metadata_output,
            minimum_events_per_bin=args.minimum_events_per_bin,
            minimum_effective_collision_count=args.minimum_effective_collision_count,
            maximum_relative_standard_error=args.maximum_relative_standard_error,
            minimum_exposure_mm=args.minimum_exposure_mm,
            maximum_zero_rate_upper_per_mm=args.maximum_zero_rate_upper_per_mm,
            maximum_energy_grid_gap_mevu=args.maximum_energy_grid_gap_mevu,
            provenance=provenance,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"compiled CINEL02 rate table: {len(result['samples'])} samples, "
        f"{len(result['groups'])} projectile/target groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
