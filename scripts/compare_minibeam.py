#!/usr/bin/env python3
"""Paired FP32 minibeam regression/timing, with isolated outputs and fixed seeds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ("[urban-v2-steps]", "[unified-em-audit]", "COPPER_PROTON_LOSS_QUALITY",
          "AIR_IONIZATION_CLOCK", "[all-ion-elastic]", "[minibeam-water] delta")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_fields(value, prefix=""):
    """Compare discrete counters, not fixed-point/rounded energy aggregates."""
    # This SchneiderFloatSlot-backed double can serialize without a decimal
    # point; its JSON representation must not turn it into a discrete count.
    if ("MeV" in prefix or "_sums" in prefix
            or prefix.endswith(("/primary_selected_energy_mismatch_sum",
                                "/secondary_selected_energy_mismatch_sum"))):
        return {}
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(integer_fields(child, prefix + "/" + key))
        return result
    if isinstance(value, list):
        result = {}
        for key, child in enumerate(value):
            result.update(integer_fields(child, prefix + "/" + str(key)))
        return result
    return {prefix: value} if type(value) is int else {}


def run(executable, preset, directory, histories, seed, shards, phase, short_search=None, env_extra=None):
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("data", "inputs"):
        (directory / name).symlink_to(ROOT / name, target_is_directory=True)
    config = yaml.safe_load((ROOT / "config" / (preset + ".yaml")).read_text())
    config.update(number_of_histories=histories, random_seed=seed)
    if config.get("tpsSource") and config.get("tps_spot_weight_mode") == "histories":
        # Absolute spot weights override number_of_histories. This harness
        # supports the shipped single-spot carbon minibeam, not plan resampling.
        with (ROOT / config["tps_spots_file"]).open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        if len(rows) != 1:
            raise ValueError("Use an explicit matched plan for multi-spot timing")
        rows[0]["weight"] = str(histories)
        with (directory / "spots.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        config.update(tps_spots_file="spots.csv", tps_histories_scale=1.0)
    if config.get("enable_charged_origin_voxel_scoring"):
        config["charged_origin_voxel_mhd_output_prefix"] = "out/case/components"
    (directory / "case.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = {k: v for k, v in os.environ.items() if not k.startswith("CARBON_")}
    compiler = Path(os.environ.get("CXX", "/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx"))
    env["LD_LIBRARY_PATH"] = str(compiler.parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(CARBON_RUNTIME_BREAKDOWN="1", OPENBLAS_NUM_THREADS="1")
    if shards is not None:
        env["CARBON_MINIBEAM_WATER_AUDIT_SHARDS"] = str(shards)
    if short_search is not None:
        env["CARBON_MINIBEAM_SHORT_SLIT_SEARCH"] = str(short_search)
    env.update(env_extra or {})
    command = [str(executable), "--config", "case.yaml"]
    if phase:
        command += ["--minibeam-phase-space-output", "entry.csv"]
    metadata = dict(command=command, binary_sha256=sha(executable),
                    config_sha256=sha(directory / "case.yaml"), histories=histories,
                    seed=seed, shards=shards, short_search=short_search, preset=preset,
                    environment={k:v for k,v in env.items() if k.startswith("CARBON_")})
    save(directory / "execution.json", metadata)
    with (directory / "gpu_telemetry.csv").open("w") as telemetry:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                                    "--format=csv,noheader", "-l", "1"], stdout=telemetry, stderr=subprocess.DEVNULL)
        start = time.perf_counter()
        try:
            with (directory / "gpu.log").open("w") as log:
                process = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
            wall = time.perf_counter() - start
        finally:
            monitor.terminate()
            monitor.wait()
    metadata.update(exit_code=process.returncode, wall_seconds=wall, histories_per_second=None)
    save(directory / "execution.json", metadata)
    quality_path = directory / "out/case/quality_report.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    metadata.update(accepted=process.returncode == 0 and quality.get("accepted", False)
                    and quality.get("queue_overflow_count") == 0, quality=quality)
    text = (directory / "gpu.log").read_text()
    counts = re.findall(r"^Histories: (\d+)$", text, re.M)
    metadata["actual_histories"] = int(counts[-1]) if counts else None
    metadata["accepted"] = metadata["accepted"] and metadata["actual_histories"] == histories
    # Frozen experiments retain these switches; current production source
    # does not. Never silently label an ignored environment option a control.
    features_reported = (
        (shards is None or f"[water-response-audit] shards={shards}" in text)
        and (short_search is None or f"[short-step-slit-search] enabled={short_search}" in text))
    metadata["requested_features_reported"] = features_reported
    metadata["accepted"] = metadata["accepted"] and features_reported
    match = re.search(r"beamline_s=([\d.eE+-]+) water_s=([\d.eE+-]+)", text)
    if match:
        metadata.update(beamline_seconds=float(match[1]), medium_seconds=float(match[2]))
    metadata["stage_lines"] = [line for line in text.splitlines() if line.startswith(
        ("[runtime-stage]", "[electron-queue-time]", "[primary-split-time]", "[water-response-audit]"))]
    if metadata["accepted"]:
        metadata["histories_per_second"] = histories / wall
    save(directory / "execution.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("preset", "seed", "shards", "wall_seconds", "histories_per_second", "accepted")}), flush=True)
    if not metadata["accepted"]:
        raise RuntimeError("Runtime rejected: " + str(directory))
    return metadata


def compare(a, b, phase):
    output = dict(integer_mismatches=[], audits_equal=False)
    for name in ("energy_ledger.json", "neutron_transport.json"):
        pa, pb = a / "out/case" / name, b / "out/case" / name
        if pa.exists() != pb.exists():
            output["integer_mismatches"].append(name + ": availability changed")
        elif pa.exists():
            x, y = integer_fields(json.loads(pa.read_text())), integer_fields(json.loads(pb.read_text()))
            for key in sorted(x.keys() | y.keys()):
                if x.get(key) != y.get(key):
                    output["integer_mismatches"].append([name + key, x.get(key), y.get(key)])
    audits = lambda p: [s for s in (p / "gpu.log").read_text().splitlines() if s.startswith(AUDITS)]
    output["audits_equal"] = audits(a) == audits(b)
    x = np.memmap(a / "out/case/dose.raw", dtype=np.float32, mode="r")
    y = np.memmap(b / "out/case/dose.raw", dtype=np.float32, mode="r")
    if x.shape != y.shape:
        raise RuntimeError("Dose shapes differ")
    total = delta = peak = maximum = 0.0
    for i in range(0, x.size, 1000000):
        u, v = x[i:i+1000000].astype(float), y[i:i+1000000].astype(float)
        if not (np.isfinite(u).all() and np.isfinite(v).all() and (u >= 0).all() and (v >= 0).all()):
            raise RuntimeError("Invalid dose values")
        total += u.sum()
        delta += abs(u-v).sum()
        peak = max(peak, float(u.max(initial=0)))
        maximum = max(maximum, float(abs(u-v).max(initial=0)))
    output.update(dose_relative_L1=delta / total if total else 0,
                  dose_max_difference_over_peak=maximum / peak if peak else 0)
    if phase:
        output["entry_bytewise_equal"] = (a / "entry.csv").read_bytes() == (b / "entry.csv").read_bytes()
    output["exact_checks_pass"] = (output["audits_equal"] and not output["integer_mismatches"]
                                   and output.get("entry_bytewise_equal", True))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--histories", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[100113001, 100113002, 100113003])
    p.add_argument("--reference-shards", type=int, choices=[1, 64], help="Frozen audit-shard experiment only")
    p.add_argument("--candidate-shards", type=int, choices=[1, 64], help="Frozen audit-shard experiment only")
    p.add_argument("--baseline-short-search", type=int, choices=[0, 1], help="Frozen geometry experiment only")
    p.add_argument("--candidate-short-search", type=int, choices=[0, 1], help="Frozen geometry experiment only")
    p.add_argument("--phase", action="store_true")
    p.add_argument("--reverse", action="store_true", help="Start with candidate, then alternate order")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for index, seed in enumerate(args.seeds):
        results = {}
        for arm in (["baseline", "candidate"] if (index + args.reverse) % 2 == 0 else ["candidate", "baseline"]):
            executable = getattr(args, arm).resolve()
            shards = args.reference_shards if arm == "baseline" else args.candidate_shards
            results[arm] = run(executable, args.preset, args.output / f"p{index}_{arm}",
                               args.histories, seed, shards, args.phase,
                               getattr(args, arm + "_short_search"))
        checks = compare(args.output / f"p{index}_baseline", args.output / f"p{index}_candidate", args.phase)
        pair = dict(seed=seed, baseline=results["baseline"], candidate=results["candidate"], checks=checks,
                    throughput_gain=results["baseline"]["wall_seconds"] / results["candidate"]["wall_seconds"] - 1)
        pairs.append(pair)
        save(args.output / "pairs.json", pairs)
        print(json.dumps(dict(seed=seed, gain=pair["throughput_gain"], checks=checks)), flush=True)
        if not checks["exact_checks_pass"]:
            raise RuntimeError("Exact regression checks failed; inspect pairs.json")
    baseline = sum(x["baseline"]["wall_seconds"] for x in pairs)
    candidate = sum(x["candidate"]["wall_seconds"] for x in pairs)
    save(args.output / "summary.json", dict(preset=args.preset, histories_per_arm=args.histories*len(pairs),
         baseline_histories_per_second=args.histories*len(pairs)/baseline,
         candidate_histories_per_second=args.histories*len(pairs)/candidate,
         throughput_gain=baseline/candidate-1, pair_gains=[x["throughput_gain"] for x in pairs],
         checks=[x["checks"] for x in pairs]))


if __name__ == "__main__":
    main()
