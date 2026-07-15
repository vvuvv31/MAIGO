#!/usr/bin/env python3
"""Upload and run 400 MeV/u cascade jobs on remote TOPAS host."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
HOST = "192.168.31.5"
USER = "v"
REMOTE_TOPAS = "/home/v/gpu/validation/topas"
LOCAL_TOPAS = Path("validation/topas")

UPLOAD = [
    "carbon_400MeVu_water.txt",
    "carbon_400MeVu_water_cascade_reactions.txt",
    "carbon_400MeVu_water_cascade_reactions_smoke.txt",
    "carbon_400MeVu_water_cascade_reactions_development_remote.txt",
    "run_cascade_e400_remote.sh",
]


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=20,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def main() -> int:
    if not PASSWORD:
        print("Set REMOTE_TOPAS_PASSWORD", file=sys.stderr)
        return 2
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode not in {"upload", "smoke", "development", "status", "fetch"}:
        print("Usage: _remote_cascade_e400.py [upload|smoke|development|status|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_cascade_e400_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *cascade_e400*|*carbon_400MeVu_water_cascade*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/cascade_e400* 2>/dev/null || true
tail -n 15 /home/v/gpu/validation/topas/output/cascade-e400-development_nohup.log 2>/dev/null || true
grep -E 'START|DONE|Total number of histories|Results have been written|Elapsed times' \
  /home/v/gpu/validation/topas/output/cascade-e400-development_topas.log 2>/dev/null | tail -20 || true
""",
            timeout=40,
        )
        print(stdout.read().decode())
        client.close()
        return 0

    if mode == "fetch":
        out = Path("validation/topas/output")
        out.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        for name in (
            "cascade_e400_smoke_reactions.phsp",
            "cascade_e400_smoke_reactions.header",
            "cascade-e400-smoke_topas.log",
            "cascade_e400_development_reactions.phsp",
            "cascade_e400_development_reactions.header",
            "cascade-e400-development_topas.log",
            "cascade-e400-development_nohup.log",
        ):
            remote = f"{REMOTE_TOPAS}/output/{name}"
            local = out / name
            try:
                sftp.get(remote, str(local))
                print("fetched", name, local.stat().st_size, flush=True)
            except OSError as exc:
                print("missing", name, exc, flush=True)
        sftp.close()
        client.close()
        return 0

    if mode == "smoke":
        print("running smoke...", flush=True)
        _, stdout, stderr = client.exec_command(
            "cd ~/gpu && bash validation/topas/run_cascade_e400_remote.sh smoke",
            timeout=3600,
            get_pty=True,
        )
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line, end="", flush=True)
        print(stderr.read().decode(), end="")
        code = stdout.channel.recv_exit_status()
        client.close()
        return code

    # development detached
    sftp = client.open_sftp()
    launcher = """#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu
echo "[$(date -Is)] START cascade e400 development" | tee validation/topas/output/cascade-e400-development_nohup.log
bash validation/topas/run_cascade_e400_remote.sh development \
  >> validation/topas/output/cascade-e400-development_nohup.log 2>&1
echo "[$(date -Is)] DONE cascade e400 development" | tee -a validation/topas/output/cascade-e400-development_nohup.log
"""
    with sftp.file(f"{REMOTE_TOPAS}/output/run_cascade_e400_development.sh", "w") as handle:
        handle.write(launcher)
    sftp.chmod(f"{REMOTE_TOPAS}/output/run_cascade_e400_development.sh", 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && setsid bash validation/topas/output/run_cascade_e400_development.sh "
        "</dev/null >/dev/null 2>&1 & sleep 3; "
        "for pid in /proc/[0-9]*; do "
        "cmd=$(tr '\\0' ' ' < \"$pid/cmdline\" 2>/dev/null || true); "
        "case \"$cmd\" in *cascade_e400*|*carbon_400MeVu_water_cascade*|*opentopas-extension-install/bin/topas*) "
        "echo \"${pid##*/} $cmd\";; esac; done; "
        "tail -n 8 validation/topas/output/cascade-e400-development_nohup.log 2>/dev/null || true",
        timeout=40,
    )
    print(stdout.read().decode())
    print(stderr.read().decode())
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
