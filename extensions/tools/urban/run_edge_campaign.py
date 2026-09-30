"""Run only on the allocated cluster node; bounded all-ion active-step extraction."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "mfp_edges_v4"
SPECIES = json.loads((ROOT / "src/species.json").read_text())["ions"]
MATERIALS = ["Water_75eV", "M0", "M1", "M8", "M20"]


def write_status(value):
    value["updated_at_unix"] = time.time()
    temporary = CAMPAIGN / "reference_status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(CAMPAIGN / "reference_status.json")


def run_one(phase, z, a, nodes, offset, env):
    directory = CAMPAIGN / phase / ("z%d_a%d" % (z, a))
    log_path = CAMPAIGN / "logs" / ("%s_z%d_a%d.log" % (phase, z, a))
    if directory.exists() or log_path.exists():
        raise RuntimeError("Refusing to overwrite reference case: " + str(directory))
    started = time.time()
    with log_path.open("wb") as log:
        process = subprocess.run(
            [str(ROOT / "bin/extract_active_urban_mfp_edges_v4"), str(directory),
             str(z), str(a), str(nodes), str(offset)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    log = log_path.read_text(errors="replace")
    success = process.returncode == 0 and "ACTIVE_URBAN_ION_CONTEXT=PASS" in log
    files = {}
    if success:
        for material in [m+suffix for m in MATERIALS for suffix in ["", "_vectors", "_mfp", "_mfp_edges"]]:
            path = directory / (material + ".csv")
            if not path.exists():
                success = False
                break
            files[material] = {"path": str(path), "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = dict(z=z, a=a, phase=phase, nodes=nodes, offset=offset,
                  returncode=process.returncode, accepted=success,
                  elapsed_s=time.time()-started, files=files, log=str(log_path))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    CAMPAIGN.mkdir(exist_ok=False)
    (CAMPAIGN / "logs").mkdir()
    if not os.uname().nodename.startswith("cpu188"):
        raise RuntimeError("The reference campaign must run on allocated node cpu188")
    affinity = len(os.sched_getaffinity(0))
    workers = min(52, affinity, 56)
    if affinity < 2:
        raise RuntimeError("Request the assigned CPU allocation before starting the campaign")
    # Each worker inherits this limit: even 52 workers cannot exceed 208 GiB
    # of address space; the srun step additionally requests 64 GiB RAM.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    env = dict(os.environ)
    env.update(json.loads((ROOT / "geant4_data_env.json").read_text()))
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    state = dict(status="RUNNING", host=os.uname().nodename, job_id="2022978",
                 started_at_unix=time.time(), workers=workers,
                 species_count=len(SPECIES), material_count=len(MATERIALS),
                 completed=[], failed=[])
    try:
        for phase, nodes, offset in [("edges", 33, 0)]:
            state["phase"] = phase
            write_status(state)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_one, phase, z, a, nodes, offset, env)
                           for z, a in SPECIES]
                for future in as_completed(futures):
                    result = future.result()
                    state["completed"].append(result)
                    if not result["accepted"]:
                        state["failed"].append({"phase": phase, "z": result["z"],
                                                "a": result["a"], "log": result["log"]})
                    write_status(state)
                    print(phase, result["z"], result["a"], result["accepted"], flush=True)
            if state["failed"]:
                raise RuntimeError("Reference phase failed; later phases were not started")
        state["status"] = "COMPLETE"
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = str(error)
        raise
    finally:
        write_status(state)


if __name__ == "__main__":
    main()
