#!/usr/bin/env python3
"""Generate reproducible bounded-output electron-response shards.

Each shard records at most 12 histories in TOPAS binary ntuple format into
its own directory. Completion is followed by verification then aggregation;
this tool never launches an unbounded single-file campaign.

Checks planned histories/disk/CPU/memory budgets before writing shards.
It does NOT monitor live shared allocations, disk growth or completion.
Submission stays disabled here; use the separate monitored executor.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import math
import shlex
from pathlib import Path

MAX_HISTORIES_PER_SHARD = 12
# Section-0 v3 pilot: up to 50.5 MiB raw/history plus ~61 MiB dose CSV.
# Planning extrapolation, NOT a hard stochastic upper bound or disk quota.
def estimated_shard_bytes(histories):
    if not isinstance(histories, int) or not 1 <= histories <= MAX_HISTORIES_PER_SHARD:
        raise ValueError("Invalid bounded shard histories")
    return (128 + 64*histories) * 1024**2


MAX_TOTAL_CPUS = 192
MAX_TOTAL_MEM_GB = 160.0

# Step04 record metadata: TOPAS parameters captured verbatim when present in
# the effective include chain). Absent keys are recorded as null with a reason,
# never guessed.
TRACKED_CONFIG_KEYS = [
    "So/Beam/BeamParticle", "So/Beam/BeamEnergy", "So/Beam/BeamEnergySpread",
    "So/Beam/Component", "Ge/Slab/Material", "Ge/Slab/HLX", "Ge/Slab/HLY",
    "Ge/Slab/HLZ", "Ge/Slab/TransZ", "Ph/Default/Modules",
    "Sc/Dose3D/XBins", "Sc/Dose3D/YBins", "Sc/Dose3D/ZBins",
    "Sc/Dose3D/Quantity", "Ts/Seed",
]


def extract_config_metadata(text):
    record = {}
    for key in TRACKED_CONFIG_KEYS:
        matches = re.findall(rf"^[a-z]+:{re.escape(key)}\s*=\s*(.+?)\s*$",
                             text, re.MULTILINE)
        record[key] = matches[-1].strip() if matches else None
    return record


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_config_inputs(path, stack=()):
    """Pin the complete include graph; cycles and missing includes are errors."""
    path = Path(path).resolve()
    if path in stack:
        raise ValueError("Cyclic TOPAS includeFile")
    source = path.read_text()
    files, parts = {str(path): sha256_file(path)}, []
    for value in re.findall(r"^\s*includeFile\s*=\s*(.*?)\s*$", source, re.MULTILINE):
        for item in shlex.split(value, comments=True):
            child = Path(item)
            if not child.is_absolute():
                child = path.parent/child
            child_files, text = collect_config_inputs(child, (*stack, path))
            files.update(child_files)
            parts.append(text)
    parts.append(source)
    return files, "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, required=True,
                   help="Base TOPAS config included by every shard")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--histories-per-shard", type=int, default=12)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--cpus-per-shard", type=int, default=2)
    p.add_argument("--mem-gb-per-shard", type=float, default=4.0)
    p.add_argument("--disk-budget-gib", type=float, default=5.0)
    p.add_argument("--topas-exe", type=Path, default=Path("/home/wuwei/topas/topas-build/topas"))
    p.add_argument("--scorer-cc", type=Path,
                   default=Path("startup/extensions/CarbonElectronDepositNtuple.cc"))
    p.add_argument("--scorer-hh", type=Path,
                   default=Path("startup/extensions/CarbonElectronDepositNtuple.hh"))
    p.add_argument("--case-prefix", default="electron_response")
    p.add_argument("--record-schema-version", type=int, default=3,
                   help="Explicit observer schema 2 or 3; legacy v1 remains read-only")
    p.add_argument("--submit", action="store_true",
                   help="Refused here; use execute_electron_response_campaign.py for monitored execution")
    a = p.parse_args()
    if a.record_schema_version not in (2, 3):
        raise ValueError("record-schema-version must be 2 or 3")
    scorer_name = "CarbonElectronDepositNtupleV3" if a.record_schema_version == 3 else "CarbonElectronDepositNtuple"
    if a.record_schema_version == 3:
        if a.scorer_cc == Path("startup/extensions/CarbonElectronDepositNtuple.cc"):
            a.scorer_cc = Path("startup/extensions/CarbonElectronDepositNtupleV3.cc")
        if a.scorer_hh == Path("startup/extensions/CarbonElectronDepositNtuple.hh"):
            a.scorer_hh = Path("startup/extensions/CarbonElectronDepositNtupleV3.hh")
    if a.shards <= 0 or a.histories_per_shard <= 0:
        raise ValueError("Positive shard count and histories required")
    if (a.cpus_per_shard <= 0 or not math.isfinite(a.mem_gb_per_shard)
            or a.mem_gb_per_shard <= 0 or not math.isfinite(a.disk_budget_gib)
            or a.disk_budget_gib <= 0):
        raise ValueError("CPU/memory/disk budgets must be positive and finite")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.case_prefix):
        raise ValueError("case-prefix must be a simple identifier, not a path or shell text")
    if a.seed_start <= 0 or a.seed_start + a.shards - 1 > 2147483647:
        raise ValueError("Seeds must be positive signed 32-bit integers")
    if a.submit:
        raise ValueError("Submission disabled in generator; use execute_electron_response_campaign.py for monitored local-Slurm execution")
    a.output_dir = a.output_dir.resolve()
    if a.histories_per_shard > MAX_HISTORIES_PER_SHARD:
        raise ValueError(
            f"histories-per-shard {a.histories_per_shard} exceeds bounded limit "
            f"{MAX_HISTORIES_PER_SHARD}; split into smaller shards")
    if not a.base_config.is_file():
        raise ValueError(f"Missing base config {a.base_config}")
    total_cpus = a.shards * a.cpus_per_shard
    total_mem = a.shards * a.mem_gb_per_shard
    if total_cpus > MAX_TOTAL_CPUS:
        raise ValueError(f"Requested {total_cpus} CPUs exceeds {MAX_TOTAL_CPUS} budget")
    if total_mem > MAX_TOTAL_MEM_GB:
        raise ValueError(f"Requested {total_mem:.1f} GB exceeds {MAX_TOTAL_MEM_GB} GB budget")
    estimated_bytes = a.shards * estimated_shard_bytes(a.histories_per_shard)
    budget_bytes = int(a.disk_budget_gib * 1024**3)
    existing_bytes = sum(p.stat().st_size for p in a.output_dir.rglob("*") if p.is_file()) if a.output_dir.exists() else 0
    estimated_bytes += existing_bytes
    if estimated_bytes > budget_bytes:
        raise ValueError(
            f"Estimated {estimated_bytes/1024**3:.2f} GiB exceeds disk budget "
            f"{a.disk_budget_gib:.2f} GiB for {a.shards} shards; reduce shards or "
            f"raise the budget explicitly. No shard submitted.")
    config_sha = sha256_file(a.base_config)
    topas_sha = sha256_file(a.topas_exe) if a.topas_exe.is_file() else None
    if topas_sha is None:
        raise ValueError(f"TOPAS executable not found: {a.topas_exe}")
    scorer_shas = {}
    for label, path in (("cc", a.scorer_cc), ("hh", a.scorer_hh)):
        if not path.is_file():
            raise ValueError(f"Scorer source missing: {path}")
        scorer_shas[label] = sha256_file(path)
    input_files, base_text = collect_config_inputs(a.base_config)
    for source_path in (a.topas_exe, a.scorer_cc, a.scorer_hh,
                        Path(__file__).resolve().parent/"analyze_electron_deposit_steps.py"):
        input_files[str(source_path.resolve())] = sha256_file(source_path)
    if (a.output_dir/"campaign_manifest.json").exists():
        raise ValueError("Campaign manifest already exists; refuse overwrite")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config_record = extract_config_metadata(base_text)
    topas_version, geant4_version = None, None
    try:
        proc = subprocess.run([a.topas_exe.as_posix(), "--version"],
                              capture_output=True, text=True, timeout=120)
        topas_version = proc.stdout.strip() or None
    except (OSError, ValueError, subprocess.SubprocessError):
        topas_version = None
    manifest_shards = []
    for index in range(a.shards):
        seed = a.seed_start + index
        case_id = f"{a.case_prefix}_shard{index:03d}_seed{seed}"
        shard_dir = a.output_dir / case_id
        if shard_dir.exists():
            raise ValueError(f"Shard directory already exists (refuse overwrite): {shard_dir}")
        shard_dir.mkdir(parents=True)
        case_config = shard_dir / "case.txt"
        case_config.write_text(
            f"includeFile = {a.base_config.resolve()}\n"
            f"i:So/Beam/NumberOfHistoriesInRun = {a.histories_per_shard}\n"
            f"Ts/NumberOfThreads = {a.cpus_per_shard}\n"
            f"i:Ts/Seed = {seed}\n"
            f"s:Sc/ElectronDeposit/OutputType = \"Binary\"\n"
            f"s:Sc/ElectronDeposit/Quantity = \"{scorer_name}\"\n"
            f"s:Sc/ElectronDeposit/OutputFile = \"{(shard_dir/'steps').as_posix()}\"\n"
            f"s:Sc/Dose3D/OutputFile = \"{(shard_dir/'dose').as_posix()}\"\n")
        slurm = shard_dir / "run.slurm"
        slurm.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --job-name={case_id}\n"
            "#SBATCH --partition=compute\n"
            "#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task={a.cpus_per_shard}\n"
            f"#SBATCH --mem={math.ceil(a.mem_gb_per_shard*1024)}M\n"
            f"#SBATCH --output={(shard_dir/'slurm.log').as_posix()}\n"
            f"#SBATCH --error={(shard_dir/'slurm.err').as_posix()}\n"
            "set -euo pipefail\n"
            f"{shlex.quote(str(a.topas_exe.resolve()))} {shlex.quote(str(case_config.resolve()))}\n")
        metadata = {
            "case_id": case_id,
            "seed": seed,
            "histories_requested": a.histories_per_shard,
            "histories_actual": None,
            "slurm_job_id": None,
            "record_schema_version": a.record_schema_version,
            "topas_version": topas_version,
            "geant4_version": geant4_version,
            "geant4_version_status": ("not captured: TOPAS exposes only its own "
                                      "version string; null, not guessed"),
            "physics_modules": config_record.get("Ph/Default/Modules"),
            "production_cuts": None,
            "production_cuts_status": ("not captured: no production-cuts key in "
                                       "base config; null, not guessed"),
            "step_limits": None,
            "step_limits_status": ("not captured: no step-limit key in base "
                                   "config; null, not guessed"),
            "config_tracked_keys": config_record,
            "config_sha256": config_sha,
            "config_path": str(a.base_config.resolve()),
            "case_config_path": str(case_config.resolve()),
            "topas_exe": str(a.topas_exe.as_posix()),
            "topas_exe_sha256": topas_sha,
            "scorer_cc_sha256": scorer_shas["cc"],
            "scorer_hh_sha256": scorer_shas["hh"],
            "raw_steps": str((shard_dir / "steps.phsp").as_posix()),
            "raw_dose": str((shard_dir / "dose.csv").as_posix()),
            "completion_status": "pending",
            "analysis_status": "pending",
        }
        (shard_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        input_files[str(case_config.resolve())] = sha256_file(case_config)
        input_files[str(slurm.resolve())] = sha256_file(slurm)
        manifest_shards.append({"case_id": case_id, "seed": seed,
                                "dir": str(shard_dir.resolve())})
    manifest = {
        "input_files": input_files,
        "effective_config": base_text,
        "case_prefix": a.case_prefix,
        "shard_count": a.shards,
        "histories_per_shard": a.histories_per_shard,
        "histories_total_requested": a.shards * a.histories_per_shard,
        "seed_start": a.seed_start,
        "cpus_per_shard": a.cpus_per_shard,
        "mem_gb_per_shard": a.mem_gb_per_shard,
        "total_cpus_requested": total_cpus,
        "total_mem_gb_requested": total_mem,
        "disk_budget_gib": a.disk_budget_gib,
        "disk_budget_bytes": budget_bytes,
        "estimated_bytes": estimated_bytes,
        "estimated_bytes_per_shard": estimated_shard_bytes(a.histories_per_shard),
        "estimated_gib": estimated_bytes / 1024**3,
        "config_sha256": config_sha,
        "config_tracked_keys": config_record,
        "record_schema_version": a.record_schema_version,
        "topas_version": topas_version,
        "topas_exe_sha256": topas_sha,
        "scorer_cc_sha256": scorer_shas["cc"],
        "scorer_hh_sha256": scorer_shas["hh"],
        "submit_requested": bool(a.submit),
        "status": "planning_only; not resource-monitored or completion-verified",
        "shards": manifest_shards,
    }
    (a.output_dir / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
