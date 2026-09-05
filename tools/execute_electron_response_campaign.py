#!/usr/bin/env python3
"""Serial local-Slurm executor for pinned electron-response pilot campaigns.

No GPU jobs. Polling limits sampled disk growth (not a filesystem quota).
A shared advisory lock serializes cooperating executors. Other launchers can
race resource checks; on observing a budget breach only our own job is stopped.
No automatic resubmission after an ambiguous submission; no data deletion.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_electron_response_diagnostic import sha256_file, estimated_shard_bytes
from audit_schneider_response_scope import schneider_identity

DATA_ROOT = Path("/mnt/sda/wuwei")
MAX_CPUS, MAX_MEM = 192, 160*1024**3


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr.strip()} {result.stdout.strip()}")
    return result.stdout.strip()


def memory_bytes(value, cpus, nodes):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT])([cn]?)", value, re.I)
    if not m or float(m[1]) <= 0:
        raise ValueError("Unknown/unbounded Slurm memory request: " + value)
    factor = cpus if m[3].lower() == "c" else nodes
    return math.ceil(float(m[1])*1024**("KMGT".index(m[2].upper())+1)*factor)


def queue_usage(text):
    cpus = mem = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        jid, c, n, m = line.strip().split("|")
        c, n = int(c), int(n)
        if c <= 0 or n <= 0:
            raise ValueError("Invalid Slurm resource row")
        cpus += c
        mem += memory_bytes(m.strip(), c, n)
    return cpus, mem


def disk_bytes(root):
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in campaign output; disk scope is ambiguous")
        if path.is_file():
            total += path.stat().st_size
    return total


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def verify_inputs(manifest):
    pins = manifest.get("input_files")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Missing input/include-chain pins; regenerate campaign")
    for path, expected in pins.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("Input SHA mismatch: " + path)


def layout(manifest, hu):
    params = {}
    for line in manifest["effective_config"].splitlines():
        m = re.match(r"^(?:[a-z]+:)?([^=]+?)\s*=\s*(.*?)\s*$", line)
        if m and not m[1].startswith("#"):
            params[m[1].strip()] = m[2].strip().strip('"')
    def mm(key, default=None):
        value = params.get(key, default)
        if value is None:
            raise ValueError("Missing geometry parameter " + key)
        parts = str(value).split()
        if len(parts) != 2 or parts[1] not in ("mm", "cm", "m"):
            raise ValueError("Unsupported length " + key)
        return float(parts[0])*{"mm":1, "cm":10, "m":1000}[parts[1]]
    if params.get("Sc/Dose3D/Quantity") != "DoseToMedium" or params.get("Sc/Dose3D/Component") != "Slab":
        raise ValueError("Executor requires Slab 3D DoseToMedium")
    if params.get("Ge/Slab/Type") != "TsBox":
        raise ValueError("Executor requires a homogeneous TsBox slab")
    for axis in "XYZ":
        if params.get("Ge/Slab/Rot"+axis, "0 deg") != "0 deg":
            raise ValueError("Rotated slab unsupported by terminal audit")
    expected = "PatientTissueFromHU" + ("Negative"+str(-hu) if hu < 0 else str(hu))
    if params.get("Ge/Slab/Material") != expected:
        raise ValueError("Declared HU does not match actual slab material")
    bounds, volume = [], 1.0
    for axis in "XYZ":
        half = mm("Ge/Slab/HL"+axis)
        centre = mm("Ge/Slab/Trans"+axis, "0 mm")
        bins = int(params["Sc/Dose3D/"+axis+"Bins"])
        if not math.isfinite(half) or half <= 0 or bins <= 0 or not math.isfinite(centre):
            raise ValueError("Invalid slab/scorer dimensions")
        bounds += [centre-half, centre+half]
        volume *= 2*half/bins
    return bounds, volume


def validate_report(result, identity, histories):
    """Require measured material identity, not just the configured material name."""
    if result.get("record_schema_version") != 3 or result.get("parent_step_binding_verified") is not True:
        raise ValueError("Generating-step binding/schema did not pass")
    if result.get("events_observed") != histories or result.get("histories_requested") != histories:
        raise ValueError("Analyzed history count mismatch")
    rho = result.get("material_density_g_cm3")
    if (not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0
            or not math.isclose(rho, identity["density_g_cm3"], rel_tol=1e-5, abs_tol=0)):
        raise ValueError("Measured material density does not match Schneider HU formula")
    ratio = result.get("dose3d_over_steps")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or abs(ratio-1) > 1e-3:
        raise ValueError("3D dose/step energy closure failed")


def execute(campaign, hu, poll=5, timeout=1800):
    campaign = Path(campaign).resolve()
    if not campaign.is_relative_to(DATA_ROOT) or campaign.parent == DATA_ROOT:
        raise ValueError("Campaign must have a dedicated directory inside local /mnt/sda/wuwei")
    root = campaign.parent
    manifest = json.loads(campaign.read_text())
    if manifest.get("record_schema_version") != 3:
        raise ValueError("Execution requires v3 generating-step records")
    verify_inputs(manifest)
    identity = schneider_identity(Path(__file__).resolve().parents[1]/"data/HUtoMaterialSchneider.txt", hu)
    if identity["material_section"] != 0:
        raise ValueError("Density pilot requires material section 0")
    bounds, volume = layout(manifest, hu)
    analyzer = Path(__file__).resolve().parent/"analyze_electron_deposit_steps.py"
    if str(analyzer) not in manifest["input_files"]:
        raise ValueError("Analyzer must be pinned")
    budget = int(manifest["disk_budget_bytes"])
    headroom = estimated_shard_bytes(manifest["histories_per_shard"])
    need_cpu = int(manifest["cpus_per_shard"])
    need_mem = math.ceil(float(manifest["mem_gb_per_shard"])*1024)*1024**2
    if budget <= 0 or not 0 < need_cpu <= MAX_CPUS or not 0 < need_mem <= MAX_MEM:
        raise ValueError("Invalid campaign resource budget")
    deadline = time.monotonic()+timeout
    state_path = root/"execution.json"
    if state_path.exists():
        raise ValueError("Execution record exists; refuse duplicate/restart without audit")
    state = {"status":"preflight", "material_identity":identity, "jobs":[],
             "executor_sha256":sha256_file(Path(__file__)),
             "campaign_sha256":sha256_file(campaign),
             "disk_monitor":"sampled every poll; not a hard quota", "poll_seconds":poll}
    active = None
    with (DATA_ROOT/".electron-response-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(state_path, state)
        try:
            for item in manifest["shards"]:
                meta = None
                folder = Path(item["dir"]).resolve()
                if not folder.is_relative_to(root) or folder == root:
                    raise ValueError("Shard directory escapes campaign")
                meta_path = folder/"metadata.json"
                meta = json.loads(meta_path.read_text())
                if meta.get("slurm_job_id") is not None or any(folder.glob("*.phsp")):
                    raise ValueError("Shard already submitted or contains raw output")
                script = folder/"run.slurm"
                if str(script) not in manifest["input_files"]:
                    raise ValueError("Slurm script must be pinned")
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Resource wait timed out")
                    used = queue_usage(command(["squeue","--me","--noheader","--format=%i|%C|%D|%m"]))
                    if disk_bytes(root)+headroom > budget:
                        raise ValueError("Insufficient disk headroom; no new job submitted")
                    if used[0]+need_cpu <= MAX_CPUS and used[1]+need_mem <= MAX_MEM:
                        break
                    state["status"] = "waiting_resources"
                    write_json(state_path, state)
                    time.sleep(poll)
                verify_inputs(manifest)
                # Intent persists before sbatch so a crash cannot silently restart.
                state["status"] = "submission_pending"
                state["submitting"] = item["case_id"]
                write_json(state_path, state)
                job = command(["sbatch","--parsable",str(script)]).split(";")[0]
                if not re.fullmatch(r"\d+", job):
                    raise ValueError("Ambiguous sbatch response; manual reconciliation required")
                active = job
                record = {"case_id":item["case_id"],"job_id":job,"status":"submitted",
                          "resources_before":{"cpus":used[0],"memory_bytes":used[1]}}
                state["jobs"].append(record)
                meta.update(slurm_job_id=job, completion_status="submitted")
                write_json(meta_path, meta)
                write_json(state_path, state)
                print("submitted", job, item["case_id"], flush=True)
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Job monitoring timed out")
                    used = queue_usage(command(["squeue","--me","--noheader","--format=%i|%C|%D|%m"]))
                    measured = disk_bytes(root)
                    state.update(status="monitoring", disk_bytes=measured)
                    write_json(state_path, state)
                    if measured > budget or used[0] > MAX_CPUS or used[1] > MAX_MEM:
                        raise ValueError("Observed shared-resource or disk budget breach")
                    output = command(["sacct","--jobs",job,"--noheader","--parsable2",
                                      "--format=JobIDRaw,State,ExitCode"])
                    rows = [x.split("|") for x in output.splitlines() if x.split("|")[0] == job]
                    if rows:
                        status, exit_code = rows[0][1:3]
                        record.update(slurm_state=status, exit_code=exit_code)
                        if status == "COMPLETED":
                            active = None
                            if exit_code != "0:0":
                                raise ValueError("Nonzero completed-job exit code")
                            break
                        if status.split()[0].rstrip("+") in {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE"}:
                            active = None
                            raise ValueError("Slurm job failed: "+status)
                    time.sleep(poll)
                verify_inputs(manifest)
                meta["completion_status"] = "completed"
                meta["analysis_status"] = "running"
                write_json(meta_path, meta)
                report = folder/"analysis.json"
                with (folder/"analysis.log").open("w") as log:
                    subprocess.run([sys.executable,str(analyzer),"--steps",str(folder/"steps.phsp"),
                        "--dose",str(folder/"dose.csv"),"--histories",str(manifest["histories_per_shard"]),
                        "--format","topas-binary-le","--voxel-volume-mm3",str(volume),
                        "--slab-bounds-mm",*map(str,bounds),"--output",str(report)],
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                result = json.loads(report.read_text())
                validate_report(result, identity, manifest["histories_per_shard"])
                if disk_bytes(root) > budget:
                    raise ValueError("Post-analysis disk budget exceeded")
                meta.update(analysis_status="validated", histories_actual=result["events_observed"],
                            report_sha256=sha256_file(report))
                meta["raw_files"] = [{"path":result["raw_"+k],"sha256":result[k+"_sha256"],
                                     "size_bytes":Path(result["raw_"+k]).stat().st_size}
                                    for k in ("steps","header","dose")]
                write_json(meta_path, meta)
                record["status"] = "validated"
                write_json(state_path, state)
                print("validated", job, flush=True)
            state["status"] = "validated"
            write_json(state_path, state)
        except BaseException as error:
            if active is not None:
                subprocess.run(["scancel",active], capture_output=True, timeout=30)
                state["cancel_requested_job_id"] = active
            state.update(status="failed", error=str(error))
            if locals().get("meta") is not None:
                meta["error"] = str(error)
                if meta.get("completion_status") == "completed":
                    meta["analysis_status"] = "failed"
                else:
                    meta["completion_status"] = "failed_or_cancel_requested"
                write_json(meta_path, meta)
            write_json(state_path, state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--hu", type=int, required=True)
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    a = p.parse_args()
    if not 1 <= a.poll_seconds <= 30 or not 1 <= a.timeout_seconds <= 7200:
        raise ValueError("Invalid polling/deadline bounds")
    execute(a.campaign, a.hu, a.poll_seconds, a.timeout_seconds)


if __name__ == "__main__":
    main()
