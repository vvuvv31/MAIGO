#!/usr/bin/env python3
"""Independent statistical checks for the frozen C12-water MCS candidate.

This is a NumPy reference sampler, not a dose test.  It checks the analytic
Fermi--Eyges moments, the path-Poisson tail, the conditional diagnostic bridge,
long-step versus split-step composition, and covariance after rotating an
inclined incident direction.  Keep its constants synchronized with
water_c12_fermi_eyges_tail_step().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CORE_SCATTERING_ENERGY_MEV = 9.9
TAIL_RATE_PER_MM = 0.0025
TAIL_SCATTERING_ENERGY_MEV = 2.4
NUCLEON_REST_MEV = 931.49410242
WATER_DENSITY_G_CM3 = 1.0
WATER_RADIATION_LENGTH_G_CM2 = 36.083


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--energy-MeVu", type=float, default=160.0)
    parser.add_argument("--path-mm", type=float, default=20.0)
    parser.add_argument("--split-mm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def powers(energy_MeVu: float) -> tuple[float, float, float]:
    total = energy_MeVu + NUCLEON_REST_MEV
    momentum_u = np.sqrt(
        energy_MeVu * (energy_MeVu + 2.0 * NUCLEON_REST_MEV))
    beta = momentum_u / total
    momentum = 12.0 * momentum_u
    radiation_length_mm = (
        10.0 * WATER_RADIATION_LENGTH_G_CM2 / WATER_DENSITY_G_CM3)
    charge_over_beta_momentum = 6.0 / (beta * momentum)
    core = ((CORE_SCATTERING_ENERGY_MEV *
             charge_over_beta_momentum) ** 2 / radiation_length_mm)
    tail_event_variance = (
        TAIL_SCATTERING_ENERGY_MEV * charge_over_beta_momentum) ** 2
    return core, tail_event_variance, core + TAIL_RATE_PER_MM * tail_event_variance


def sample_interval(rng: np.random.Generator, samples: int, length_mm: float,
                    core_power: float,
                    tail_event_variance: float) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    theta = np.sqrt(core_power * length_mm) * rng.normal(size=samples)
    displacement = (0.5 * length_mm * theta +
                    np.sqrt(core_power * length_mm ** 3 / 12.0) *
                    rng.normal(size=samples))
    counts = rng.poisson(TAIL_RATE_PER_MM * length_mm, size=samples)
    event_history = np.repeat(np.arange(samples), counts)
    locations = rng.random(event_history.size) * length_mm
    if event_history.size:
        event_theta = np.sqrt(tail_event_variance) * rng.normal(
            size=event_history.size)
        np.add.at(theta, event_history, event_theta)
        np.add.at(displacement, event_history,
                  (length_mm - locations) * event_theta)
    return theta, displacement, counts, locations


def moments(theta: np.ndarray, displacement: np.ndarray) -> dict[str, float]:
    return {
        "angle_variance_rad2": float(np.var(theta)),
        "displacement_angle_cov_mm_rad": float(np.cov(
            displacement, theta, ddof=0)[0, 1]),
        "displacement_variance_mm2": float(np.var(displacement)),
    }


def ratios(observed: dict[str, float], expected: dict[str, float]) -> dict[str, float]:
    return {key: observed[key] / expected[key] for key in expected}


def bridge_sample(rng: np.random.Generator, theta_l: np.ndarray,
                  displacement_l: np.ndarray, length_mm: float,
                  core_power: float, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    r = fraction
    r2 = r * r
    r3 = r2 * r
    c00, c01 = r, r - 0.5 * r2
    c10, c11 = 0.5 * r2, 0.5 * r2 - r3 / 6.0
    k00, k01 = 4.0 * c00 - 6.0 * c01, -6.0 * c00 + 12.0 * c01
    k10, k11 = 4.0 * c10 - 6.0 * c11, -6.0 * c10 + 12.0 * c11
    var_theta = max(0.0, r - (k00 * c00 + k01 * c01))
    covariance = 0.5 * r2 - (k00 * c10 + k01 * c11)
    var_displacement = max(0.0, r3 / 3.0 - (k10 * c10 + k11 * c11))
    l00 = np.sqrt(var_theta)
    l10 = covariance / l00 if l00 > 0.0 else 0.0
    l11 = np.sqrt(max(0.0, var_displacement - l10 * l10))
    theta_scale = np.sqrt(core_power * length_mm)
    displacement_scale = theta_scale * length_mm
    normalized_theta = theta_l / theta_scale
    normalized_displacement = displacement_l / displacement_scale
    z0 = rng.normal(size=theta_l.size)
    z1 = rng.normal(size=theta_l.size)
    theta_t = theta_scale * (
        k00 * normalized_theta + k01 * normalized_displacement + l00 * z0)
    displacement_t = displacement_scale * (
        k10 * normalized_theta + k11 * normalized_displacement +
        l10 * z0 + l11 * z1)
    return theta_t, displacement_t


def main() -> None:
    args = arguments()
    if args.samples <= 0 or args.path_mm <= 0.0 or args.split_mm <= 0.0:
        raise ValueError("samples and path lengths must be positive")
    core_power, tail_variance, total_power = powers(args.energy_MeVu)
    expected = {
        "angle_variance_rad2": total_power * args.path_mm,
        "displacement_angle_cov_mm_rad": total_power * args.path_mm ** 2 / 2.0,
        "displacement_variance_mm2": total_power * args.path_mm ** 3 / 3.0,
    }

    rng = np.random.default_rng(args.seed)
    theta_long, displacement_long, counts, locations = sample_interval(
        rng, args.samples, args.path_mm, core_power, tail_variance)
    long_moments = moments(theta_long, displacement_long)

    segment_count = int(round(args.path_mm / args.split_mm))
    if not np.isclose(segment_count * args.split_mm, args.path_mm):
        raise ValueError("path-mm must be an integer multiple of split-mm")
    theta_split = np.zeros(args.samples)
    displacement_split = np.zeros(args.samples)
    total_split_counts = np.zeros(args.samples, dtype=np.int64)
    for _ in range(segment_count):
        segment_theta, segment_displacement, segment_counts, _ = sample_interval(
            rng, args.samples, args.split_mm, core_power, tail_variance)
        displacement_split += segment_displacement + args.split_mm * theta_split
        theta_split += segment_theta
        total_split_counts += segment_counts
    split_moments = moments(theta_split, displacement_split)

    # Core-only endpoint and its half-step bridge.  The endpoint arrays are
    # retained before/after the query to make non-perturbation explicit.
    theta_core = np.sqrt(core_power * args.path_mm) * rng.normal(size=args.samples)
    displacement_core = (
        0.5 * args.path_mm * theta_core +
        np.sqrt(core_power * args.path_mm ** 3 / 12.0) *
        rng.normal(size=args.samples))
    endpoint_checksum = [float(np.sum(theta_core)), float(np.sum(displacement_core))]
    theta_half, displacement_half = bridge_sample(
        rng, theta_core, displacement_core, args.path_mm, core_power, 0.5)
    bridge_expected = {
        "angle_variance_rad2": core_power * args.path_mm * 0.5,
        "displacement_angle_cov_mm_rad": (
            core_power * (0.5 * args.path_mm) ** 2 / 2.0),
        "displacement_variance_mm2": (
            core_power * (0.5 * args.path_mm) ** 3 / 3.0),
    }

    # Rotation check for a deliberately inclined beam.  Projecting the sampled
    # covariance back into its transverse basis must recover the local result.
    incident = np.asarray([0.3, -0.2, 0.9327379053])
    incident /= np.linalg.norm(incident)
    reference = np.asarray([1.0, 0.0, 0.0])
    basis_u = reference - np.dot(reference, incident) * incident
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(incident, basis_u)
    global_theta = theta_long[:, None] * basis_u[None, :]
    global_displacement = displacement_long[:, None] * basis_u[None, :]
    projected_theta = global_theta @ basis_u
    projected_displacement = global_displacement @ basis_u

    result = {
        "frozen_parameters": {
            "core_scattering_energy_MeV": CORE_SCATTERING_ENERGY_MEV,
            "tail_rate_per_mm": TAIL_RATE_PER_MM,
            "tail_scattering_energy_MeV": TAIL_SCATTERING_ENERGY_MEV,
        },
        "samples": args.samples,
        "energy_MeVu": args.energy_MeVu,
        "path_mm": args.path_mm,
        "analytic_moments": expected,
        "one_long_step": {
            "moments": long_moments,
            "analytic_ratios": ratios(long_moments, expected),
        },
        "split_steps": {
            "segment_mm": args.split_mm,
            "segment_count": segment_count,
            "moments": split_moments,
            "analytic_ratios": ratios(split_moments, expected),
            "long_step_ratios": ratios(split_moments, long_moments),
        },
        "poisson": {
            "expected_mean": TAIL_RATE_PER_MM * args.path_mm,
            "observed_mean_long": float(np.mean(counts)),
            "observed_variance_long": float(np.var(counts)),
            "observed_mean_split": float(np.mean(total_split_counts)),
            "event_count_long": int(np.sum(counts)),
            "location_fraction_mean": (
                float(np.mean(locations / args.path_mm)) if locations.size else None),
            "location_fraction_variance": (
                float(np.var(locations / args.path_mm)) if locations.size else None),
        },
        "half_step_core_bridge": {
            "moments": moments(theta_half, displacement_half),
            "analytic_ratios": ratios(
                moments(theta_half, displacement_half), bridge_expected),
            "endpoint_checksum_before": endpoint_checksum,
            "endpoint_checksum_after": [
                float(np.sum(theta_core)), float(np.sum(displacement_core))],
        },
        "inclined_projection": {
            "incident_direction": incident.tolist(),
            "basis_orthogonality": float(np.dot(basis_u, basis_v)),
            "projected_moments": moments(
                projected_theta, projected_displacement),
            "projected_to_local_ratios": ratios(
                moments(projected_theta, projected_displacement), long_moments),
        },
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
