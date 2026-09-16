#!/usr/bin/env python3
"""Run a plan manifest as several concurrent --plan-manifest groups.

Each group is a separate process with its own SyclTransportContext, so within a
group the validated read-only physics cache is still reused, and different groups
overlap host setup with GPU kernels (and, where registers allow, their blocks).
This is orchestration only: no physics, RNG or scoring change.

Fail-closed: the full manifest is preflighted as one group before splitting, so
cross-process output collisions and device mismatches are rejected up front; if
any group exits non-zero the remaining groups are terminated and the tool exits
non-zero. Completion is verified against the expected config set, so a worker
that exits 0 without doing its transport (for example a help/dry-run path) is
never counted as a successful shard.

Usage:
  python3 tools/plan_concurrent.py --manifest plan.txt --groups 3 \
      [--executable build/oneapi-nvidia-release/carbon_mc] [--device cuda] \
      [-- <extra carbon_mc args>]
"""
import argparse
import os
import re
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


def preflight(executable: Path, manifest: Path, device: str, extra, cwd: Path) -> None:
    """Validate the unsplit manifest once, using the same CLI overrides.

    Each worker process only sees its own group, so per-process collision checks
    cannot catch two shards in different groups writing the same file. Running
    the binary's own preflight over the full manifest restores that global check
    without loading CT/source or launching transport.
    """
    cmd = [str(executable), "--plan-manifest", str(manifest.resolve()),
           "--device", device, "--plan-preflight"] + list(extra)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(
            "plan preflight rejected the manifest; refusing to split it across "
            "processes (config/override/device/output-collision check)")



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
    extra = [x for x in a.extra if x != "--"]
    base = a.manifest.resolve().parent
    executable = a.executable.expanduser().resolve()
    preflight(executable, a.manifest, a.device, extra, base)
    groups = [[] for _ in range(min(a.groups, len(configs)))]
    for i, cfg in enumerate(configs):
        groups[i % len(groups)].append(cfg)

    tmp = Path(tempfile.mkdtemp(prefix="plan_concurrent_", dir=str(base)))
    procs = []
    started = time.monotonic()
    for g, members in enumerate(groups):
        gm = (tmp / f"group_{g}.txt").resolve()
        gm.write_text("\n".join(str(m) for m in members) + "\n")
        cmd = [str(executable), "--plan-manifest", str(gm), "--device", a.device] + extra
        log = (tmp / f"group_{g}.log").open("w")
        procs.append((g, members, subprocess.Popen(cmd, cwd=base, stdout=log,
                                                    stderr=subprocess.STDOUT), log))
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

    # Verify every expected shard completed exactly once and passed, then use the
    # binary-reported histories (not requested values) as the throughput
    # numerator. A zero-exit worker with no transport record must not count.
    done_re = re.compile(
        r"\[plan-shard-done\]\s+\d+/\d+\s+config=(.*?)\s+histories=(\d+)\s+quality=(\S+)")
    completed: dict = {}
    for g in range(len(groups)):
        text = (tmp / f"group_{g}.log").read_text(errors="replace")
        for match in done_re.finditer(text):
            cfg = os.path.realpath(match.group(1).strip().strip('"'))
            if cfg in completed:
                raise SystemExit(
                    f"duplicate completion record for {cfg}; refusing to count")
            completed[cfg] = (int(match.group(2)), match.group(3))
    expected = {os.path.realpath(str(c)) for c in configs}
    missing = expected - completed.keys()
    unexpected = set(completed) - expected
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(sorted(missing)))
        if unexpected:
            detail.append("unexpected=" + ",".join(sorted(unexpected)))
        raise SystemExit(
            "incomplete plan: no exact one-to-one completion record ("
            + "; ".join(detail) + f"); see {tmp}")
    bad_quality = {c: q for c, (_, q) in completed.items() if q != "pass"}
    if bad_quality:
        raise SystemExit("plan shards not quality=pass: " + ", ".join(
            f"{c}={q}" for c, q in sorted(bad_quality.items())))
    histories = sum(h for h, _ in completed.values())
    print(f"plan_concurrent: shards={len(configs)} groups={len(groups)} "
          f"histories={histories} wall={wall:.3f}s "
          f"throughput={histories / wall:.1f} h/s logs={tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
