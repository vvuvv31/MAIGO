#!/usr/bin/env python3
"""Run a plan manifest as several concurrent --plan-manifest groups.

Each group is a separate process with its own SyclTransportContext, so within a
group the validated read-only physics cache is still reused, and different groups
overlap host setup with GPU kernels (and, where registers allow, their blocks).
This is orchestration only: no physics, RNG or scoring change.

Fail-closed: if any group exits non-zero the remaining groups are terminated and
the tool exits non-zero, so partial/failed shards never look like a success.

Usage:
  python3 tools/plan_concurrent.py --manifest plan.txt --groups 3 \
      [--executable build/oneapi-nvidia-release/carbon_mc] [--device cuda] \
      [-- <extra carbon_mc args>]
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def read_manifest(path: Path):
    path = path.resolve()
    base = path.parent
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line)
        out.append((entry if entry.is_absolute() else base / entry).resolve())
    if not out:
        raise SystemExit("empty plan manifest: " + str(path))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--executable", type=Path,
                   default=Path("build/oneapi-nvidia-release/carbon_mc"))
    p.add_argument("--device", default="cuda")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.groups < 1:
        raise SystemExit("--groups must be >= 1")
    configs = read_manifest(a.manifest)
    groups = [[] for _ in range(min(a.groups, len(configs)))]
    for i, cfg in enumerate(configs):
        groups[i % len(groups)].append(cfg)

    base = a.manifest.resolve().parent
    executable = a.executable.expanduser().resolve()
    tmp = Path(tempfile.mkdtemp(prefix="plan_concurrent_", dir=str(base)))
    procs = []
    started = time.monotonic()
    for g, members in enumerate(groups):
        gm = (tmp / f"group_{g}.txt").resolve()
        gm.write_text("\n".join(str(m) for m in members) + "\n")
        extra = [x for x in a.extra if x != "--"]
        cmd = [str(executable), "--plan-manifest", str(gm), "--device", a.device] + extra
        log = (tmp / f"group_{g}.log").open("w")
        procs.append((g, members, subprocess.Popen(cmd, cwd=base, stdout=log,
                                                    stderr=subprocess.STDOUT), log))
    import re
    failed = None
    remaining = list(procs)
    while remaining:
        for item in list(remaining):
            g, members, proc, log = item
            rc = proc.poll()
            if rc is not None:
                remaining.remove(item)
                log.close()
                if rc != 0 and failed is None:
                    failed = g
        if failed is not None:
            for g, members, proc, log in remaining:
                proc.terminate()
            for g, members, proc, log in remaining:
                proc.wait()
                log.close()
            break
        time.sleep(0.2)
    wall = time.monotonic() - started
    if failed is not None:
        raise SystemExit(f"group {failed} failed; see {tmp}/group_{failed}.log")

    # Throughput numerator: sum the histories the binary reports as completed,
    # not the requested config values.
    histories = 0
    for g in range(len(groups)):
        text = (tmp / f"group_{g}.log").read_text(errors="replace")
        histories += sum(int(m) for m in re.findall(r"\[plan-shard-done\][^\n]*histories=(\d+)", text))
    print(f"plan_concurrent: shards={len(configs)} groups={len(groups)} "
          f"histories={histories} wall={wall:.3f}s "
          f"throughput={histories / wall:.1f} h/s logs={tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
