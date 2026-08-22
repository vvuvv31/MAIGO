#!/usr/bin/env python3
"""Validate one TOPAS thin-slab run and prepare one fluctuation grid point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from topas_ascii_ntuple import EnergyLossFluctuationRow
from topas_ascii_ntuple import read_energy_loss_fluctuation_ntuple


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_probabilities() -> list[float]:
    """Return a fixed grid resolving the sparse high-loss collision tail."""

    lower_tail = [
        0.0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04,
    ]
    central = [index / 100.0 for index in range(5, 95)]
    upper_shoulder = [index / 1_000.0 for index in range(950, 991)]
    upper_tail = [index / 10_000.0 for index in range(9_901, 9_991)]
    extreme_tail = [index / 100_000.0 for index in range(99_901, 100_001)]
    return lower_tail + central + upper_shoulder + upper_tail + extreme_tail


def parse_probabilities(text: str | None) -> list[float]:
    if text is None:
        return default_probabilities()
    try:
        probabilities = [float(field.strip()) for field in text.split(",")]
    except ValueError as error:
        raise ValueError("probabilities must be comma-separated numbers") from error
    if len(probabilities) < 3 or any(not math.isfinite(value) for value in probabilities):
        raise ValueError("probabilities require at least three finite values")
    if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
        raise ValueError("probabilities must include exact 0 and 1 endpoints")
    if any(
        current <= previous
        for previous, current in zip(probabilities, probabilities[1:])
    ):
        raise ValueError("probabilities must be strictly increasing")
    return probabilities


def empirical_quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def trapezoid_integral(probabilities: list[float], quantiles: list[float]) -> float:
    return sum(
        0.5 * (left_q + right_q) * (right_p - left_p)
        for left_p, right_p, left_q, right_q in zip(
            probabilities, probabilities[1:], quantiles, quantiles[1:]
        )
    )


def exact_type7_quantile_integral(sorted_values: list[float]) -> float:
    """Integrate the complete piecewise-linear Type-7 empirical quantile."""

    if not sorted_values:
        raise ValueError("empirical quantile sample must not be empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    weighted_sum = math.fsum(sorted_values) - 0.5 * (
        sorted_values[0] + sorted_values[-1]
    )
    return weighted_sum / (len(sorted_values) - 1)


def validate_rows(
    rows: list[EnergyLossFluctuationRow],
    projectile_z: int,
    projectile_a: int,
    material: str,
    energy_mev_per_u: float,
) -> list[float]:
    expected_entry = projectile_a * energy_mev_per_u
    entry_tolerance = max(1.0e-8, 5.0e-6 * expected_entry)
    event_keys: set[tuple[int, int]] = set()
    losses: list[float] = []

    for index, row in enumerate(rows, start=1):
        label = f"row {index} (run={row.run_id}, event={row.event_id})"
        if row.run_id < 0 or row.event_id < 0 or row.primary_track_id <= 0:
            raise ValueError(f"{label}: invalid run/event/primary track identity")
        event_key = (row.run_id, row.event_id)
        if event_key in event_keys:
            raise ValueError(f"{label}: duplicate primary history")
        event_keys.add(event_key)
        if (row.atomic_number, row.mass_number) != (projectile_z, projectile_a):
            raise ValueError(
                f"{label}: expected projectile Z{projectile_z}A{projectile_a}, "
                f"got Z{row.atomic_number}A{row.mass_number}"
            )
        if row.material_name != material or not row.material_consistent:
            raise ValueError(
                f"{label}: expected one consistent {material} slab, "
                f"got {row.material_name!r}"
            )
        if not row.completed or row.completion_status != "exited":
            raise ValueError(
                f"{label}: primary did not exit the thin slab "
                f"(completed={row.completed}, status={row.completion_status!r})"
            )
        if abs(row.entry_energy_mev - expected_entry) > entry_tolerance:
            raise ValueError(
                f"{label}: entry energy {row.entry_energy_mev:g} MeV differs from "
                f"expected {expected_entry:g} MeV"
            )
        if (
            row.exit_energy_mev < 0.0
            or row.exit_energy_mev > row.entry_energy_mev + entry_tolerance
            or row.kinetic_energy_loss_mev < 0.0
            or row.primary_local_deposit_mev < 0.0
            or row.path_length_mm <= 0.0
            or row.step_count <= 0
        ):
            raise ValueError(f"{label}: invalid energy, path length, or step count")

        reconstructed_loss = row.entry_energy_mev - row.exit_energy_mev
        closure_tolerance = max(1.0e-8, 5.0e-6 * row.entry_energy_mev)
        if abs(row.kinetic_energy_loss_mev - reconstructed_loss) > closure_tolerance:
            raise ValueError(
                f"{label}: entry-exit loss {reconstructed_loss:g} MeV does not "
                f"match recorded loss {row.kinetic_energy_loss_mev:g} MeV"
            )
        if row.primary_local_deposit_mev > row.kinetic_energy_loss_mev + closure_tolerance:
            raise ValueError(
                f"{label}: primary local deposit exceeds primary kinetic-energy loss"
            )
        losses.append(row.kinetic_energy_loss_mev)
    return losses


def prepare_point(
    input_path: Path,
    projectile_z: int,
    projectile_a: int,
    material: str,
    energy_mev_per_u: float,
    areal_density_g_per_cm2: float,
    expected_histories: int,
    probabilities: list[float],
) -> dict[str, object]:
    if projectile_z <= 0 or projectile_a < projectile_z:
        raise ValueError("projectile Z/A must satisfy Z > 0 and A >= Z")
    if not material:
        raise ValueError("material must not be empty")
    if energy_mev_per_u <= 0.0 or not math.isfinite(energy_mev_per_u):
        raise ValueError("energy must be finite and positive")
    if areal_density_g_per_cm2 <= 0.0 or not math.isfinite(areal_density_g_per_cm2):
        raise ValueError("areal density must be finite and positive")
    if expected_histories <= 0:
        raise ValueError("expected histories must be positive")

    header, rows = read_energy_loss_fluctuation_ntuple(input_path)
    if header.original_histories != expected_histories:
        raise ValueError(
            f"TOPAS header reports {header.original_histories} original histories; "
            f"expected {expected_histories}"
        )
    if header.scored_entries != expected_histories:
        raise ValueError(
            f"thin-slab scorer wrote {header.scored_entries} rows for "
            f"{expected_histories} histories"
        )
    losses = validate_rows(
        rows, projectile_z, projectile_a, material, energy_mev_per_u
    )
    sample_mean = math.fsum(losses) / len(losses)
    if sample_mean <= 0.0 or not math.isfinite(sample_mean):
        raise ValueError("sample mean energy loss must be finite and positive")

    normalized_samples = sorted(loss / sample_mean for loss in losses)
    raw_quantiles = [
        empirical_quantile(normalized_samples, probability)
        for probability in probabilities
    ]
    empirical_type7_mean = exact_type7_quantile_integral(normalized_samples)
    integral_before = trapezoid_integral(probabilities, raw_quantiles)
    if (
        empirical_type7_mean <= 0.0
        or not math.isfinite(empirical_type7_mean)
        or integral_before <= 0.0
        or not math.isfinite(integral_before)
    ):
        raise ValueError("piecewise-linear inverse CDF has an invalid mean")
    compression_mean_error = integral_before / empirical_type7_mean - 1.0
    quantiles = [value / integral_before for value in raw_quantiles]
    integral_after = trapezoid_integral(probabilities, quantiles)
    if abs(integral_after - 1.0) > 1.0e-12:
        raise RuntimeError("failed to normalize piecewise-linear inverse CDF")

    header_path = input_path if input_path.suffix == ".header" else Path(f"{input_path}.header")
    phsp_path = input_path if input_path.suffix == ".phsp" else Path(f"{input_path}.phsp")
    if input_path.suffix == ".header":
        phsp_path = input_path.with_suffix(".phsp")
    elif input_path.suffix == ".phsp":
        header_path = input_path.with_suffix(".header")

    return {
        "schema": "maigo-energy-loss-fluctuation-point-v1",
        "projectile": {"Z": projectile_z, "A": projectile_a},
        "material": material,
        "energy_MeV_per_u": energy_mev_per_u,
        "areal_density_g_per_cm2": areal_density_g_per_cm2,
        "histories": expected_histories,
        "sample_energy_loss_MeV": {
            "mean": sample_mean,
            "minimum": min(losses),
            "maximum": max(losses),
        },
        "inverse_cdf": {
            "probabilities": probabilities,
            "loss_over_mean_quantiles": quantiles,
            "empirical_quantile_method": "linear_position_p_times_n_minus_1",
            "empirical_type7_mean_exact": empirical_type7_mean,
            "piecewise_linear_mean_before_normalization": integral_before,
            "piecewise_linear_compression_mean_relative_error": compression_mean_error,
            "unit_mean_normalization_factor": 1.0 / integral_before,
            "piecewise_linear_mean_after_normalization": integral_after,
        },
        "diagnostics": {
            "mean_primary_local_deposit_MeV": math.fsum(
                row.primary_local_deposit_mev for row in rows
            ) / len(rows),
            "mean_primary_path_length_mm": math.fsum(
                row.path_length_mm for row in rows
            ) / len(rows),
            "mean_primary_step_count": math.fsum(row.step_count for row in rows) / len(rows),
        },
        "sources": {
            "header": {"path": str(header_path), "sha256": sha256(header_path)},
            "phsp": {"path": str(phsp_path), "sha256": sha256(phsp_path)},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TOPAS stem, .header, or .phsp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projectile-z", type=int, required=True)
    parser.add_argument("--projectile-a", type=int, required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--energy-mev-per-u", type=float, required=True)
    parser.add_argument("--areal-density-g-per-cm2", type=float, required=True)
    parser.add_argument("--expected-histories", type=int, required=True)
    parser.add_argument(
        "--probabilities",
        help="optional comma-separated inverse-CDF probability grid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        probabilities = parse_probabilities(args.probabilities)
        point = prepare_point(
            args.input,
            args.projectile_z,
            args.projectile_a,
            args.material,
            args.energy_mev_per_u,
            args.areal_density_g_per_cm2,
            args.expected_histories,
            probabilities,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(point, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"prepared TOPAS fluctuation point: {args.output}")


if __name__ == "__main__":
    main()
