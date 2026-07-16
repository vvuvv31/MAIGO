#!/usr/bin/env python3
"""Run a command while watching Windows GPU Adapter Memory.

If dedicated (or dedicated+shared) usage exceeds max_fraction of the given
device total, the child process is killed immediately to avoid Arc driver
lockups / host freezes.

Example:
  python validation/scripts/gpu_memory_guard.py ^
    --max-fraction 0.50 --device-mib 11869 ^
    -- build\\oneapi-windows-release-grok\\carbon_mc.exe --config config\\...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _query_gpu_adapter_bytes(counter: str) -> list[float]:
    """Return CookedValue list for a GPU Adapter Memory counter (bytes)."""
    # Escape for PowerShell single-quoted path is not needed for this pattern.
    ps = (
        f"(Get-Counter '{counter}' -ErrorAction Stop).CounterSamples | "
        "ForEach-Object { $_.CookedValue }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    values: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def gpu_dedicated_bytes() -> float:
    vals = _query_gpu_adapter_bytes(r"\GPU Adapter Memory(*)\Dedicated Usage")
    return max(vals) if vals else 0.0


def gpu_shared_bytes() -> float:
    vals = _query_gpu_adapter_bytes(r"\GPU Adapter Memory(*)\Shared Usage")
    return max(vals) if vals else 0.0


def watch_and_kill(
    proc: subprocess.Popen,
    *,
    device_bytes: float,
    max_fraction: float,
    poll_s: float,
    include_shared: bool,
    log_every_s: float,
) -> str | None:
    """Return kill reason if killed, else None when process exits on its own."""
    limit = device_bytes * max_fraction
    last_log = 0.0
    while proc.poll() is None:
        dedicated = gpu_dedicated_bytes()
        shared = gpu_shared_bytes() if include_shared else 0.0
        used = dedicated + shared
        now = time.time()
        if now - last_log >= log_every_s:
            print(
                f"[gpu-guard] dedicated={dedicated / (1024**2):.0f} MiB "
                f"shared={shared / (1024**2):.0f} MiB "
                f"limit={limit / (1024**2):.0f} MiB "
                f"({max_fraction * 100:.0f}% of {device_bytes / (1024**2):.0f} MiB)",
                flush=True,
            )
            last_log = now
        if used > limit:
            reason = (
                f"GPU memory {used / (1024**2):.0f} MiB exceeded "
                f"{max_fraction * 100:.0f}% limit {limit / (1024**2):.0f} MiB "
                f"(dedicated={dedicated / (1024**2):.0f}, shared={shared / (1024**2):.0f})"
            )
            print(f"[gpu-guard] KILL: {reason}", flush=True)
            try:
                proc.kill()
            except OSError:
                pass
            # Also kill by image name in case of child processes.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "carbon_mc.exe", "/T"],
                    capture_output=True,
                    check=False,
                )
            return reason
        time.sleep(poll_s)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.50,
        help="Kill if GPU usage exceeds this fraction of device total (default 0.50)",
    )
    parser.add_argument(
        "--device-mib",
        type=float,
        default=11869.0,
        help="Device global memory in MiB (Arc B580 ~11869). Override if needed.",
    )
    parser.add_argument("--poll-s", type=float, default=0.5)
    parser.add_argument("--log-every-s", type=float, default=5.0)
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Count shared GPU memory toward the limit (stricter)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after --  e.g. -- carbon_mc.exe --config ...",
    )
    args = parser.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command given. Use: gpu_memory_guard.py [opts] -- <cmd>...")

    if args.max_fraction <= 0.05 or args.max_fraction > 1.0:
        raise SystemExit("--max-fraction must be in (0.05, 1.0]")

    device_bytes = args.device_mib * 1024.0 * 1024.0
    # Pre-check: if already over limit, refuse to start.
    pre_d = gpu_dedicated_bytes()
    pre_s = gpu_shared_bytes() if args.include_shared else 0.0
    pre = pre_d + pre_s
    limit = device_bytes * args.max_fraction
    print(
        f"[gpu-guard] start max={args.max_fraction * 100:.0f}% "
        f"({limit / (1024**2):.0f} MiB of {args.device_mib:.0f} MiB); "
        f"now dedicated={pre_d / (1024**2):.0f} MiB",
        flush=True,
    )
    if pre > limit:
        print(
            f"[gpu-guard] REFUSE: GPU already at {pre / (1024**2):.0f} MiB > limit",
            flush=True,
        )
        return 2

    # Ensure no leftover carbon_mc holds VRAM.
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "carbon_mc.exe", "/T"],
            capture_output=True,
            check=False,
        )
        time.sleep(1.0)

    env = os.environ.copy()
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")

    proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env)
    kill_reason: list[str | None] = [None]

    def _watcher() -> None:
        kill_reason[0] = watch_and_kill(
            proc,
            device_bytes=device_bytes,
            max_fraction=args.max_fraction,
            poll_s=args.poll_s,
            include_shared=args.include_shared,
            log_every_s=args.log_every_s,
        )

    t = threading.Thread(target=_watcher, name="gpu-memory-guard", daemon=True)
    t.start()
    rc = proc.wait()
    t.join(timeout=2.0)
    if kill_reason[0] is not None:
        print(f"[gpu-guard] process killed: {kill_reason[0]}", flush=True)
        return 3
    print(f"[gpu-guard] process exited rc={rc}", flush=True)
    return int(rc) if rc is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
