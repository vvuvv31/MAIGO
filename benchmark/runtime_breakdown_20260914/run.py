"""Local RT07575 stage profiling; source/config snapshot is fixed before running.

Nsight measures kernels and transfers. Physics processes fused inside a kernel
must not be reported as independently timed by this script.
"""
from pathlib import Path
import json
import os
import re
import sqlite3
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "scratch/runtime_breakdown_20260914"
RUNNER = REPO / "benchmark/unified_em_performance_20260913/run.py"
OUTPUT = REPO / "scratch/unified_em_perf_20260913"
NSYS = "/usr/local/cuda-12.6/bin/nsys"


def main():
    rows = []
    for label, profile in [("runtime_current_1m_baseline", False),
                           ("runtime_current_1m_profile", True)]:
        folder = OUTPUT / label / "RT07575"
        status_path = folder / "status.json"
        if not status_path.exists():
            subprocess.run([sys.executable, "tools/verify_schneider_v2_1_data.py"],
                           cwd=REPO, check=True)
            env = dict(os.environ)
            env.pop("CARBON_BENCH_CONFIG_OVERRIDES", None)
            env.update(CARBON_RUNTIME_BREAKDOWN="1",
                       CARBON_BENCH_CONFIG_ROOT=str(SNAPSHOT / "inputs"),
                       CARBON_BENCH_PROFILE="1" if profile else "0")
            subprocess.run([sys.executable, str(RUNNER), label,
                            str(SNAPSHOT / "build/carbon_mc"), "RT07575"],
                           cwd=REPO, env=env, check=True)
        status = json.loads(status_path.read_text())
        if not status["complete"]:
            raise RuntimeError(f"Rejected run; inspect {folder}")
        log = (folder / "gpu.log").read_text()
        scopes = {}
        for name, seconds in re.findall(
                r"\[runtime-stage\] (\S+) seconds=([\d.eE+-]+)", log):
            scopes[name] = scopes.get(name, 0) + float(seconds)
        row = dict(label=label, status=status, host_scopes_seconds=scopes,
                   host_scope_warning="Nested scopes must not be summed",
                   grouping_seconds=sum(map(float, re.findall(
                       r"grouping_seconds=([\d.eE+-]+)", log))),
                   fused_work="CT lookup/boundaries, EM, nuclear processes, MCS and dose atomics are fused; no independent process seconds measured",
                   WET="No independent WET map stage in this configuration")
        if profile:
            db = folder / "runtime.sqlite"
            if not db.exists():
                subprocess.run([NSYS, "export", "--type", "sqlite", "--output",
                                str(db), str(folder / "runtime.nsys-rep")], check=True)
            with sqlite3.connect(db) as conn:
                row["cuda_copies"] = [dict(direction=k, seconds=s, bytes=b, count=n)
                    for k, s, b, n in conn.execute(
                        "select e.label,sum(m.end-m.start)*1e-9,sum(m.bytes),count(*) "
                        "from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e "
                        "on e.id=m.copyKind group by m.copyKind")]
                row["cuda_kernels"] = [dict(name=k, seconds=s, count=n, registers=r)
                    for k, s, n, r in conn.execute(
                        "select s.value,sum(k.end-k.start)*1e-9,count(*),avg(k.registersPerThread) "
                        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s "
                        "on s.id=k.demangledName group by k.demangledName")]
        rows.append(row)
    a, b = [r["status"] for r in rows]
    for field in ["config_sha256", "binary_sha256", "audit", "steps"]:
        if a[field] != b[field]:
            raise RuntimeError(f"Profiler comparison mismatch: {field}")
    result = dict(runs=rows,
                  profiler_elapsed_overhead_fraction=b["elapsed_s"]/a["elapsed_s"]-1,
                  warning="CUDA activities are nested inside host scopes; do not add the two timelines")
    Path(__file__).with_name("results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
