#!/usr/bin/env python3
"""Upload/run/fetch 200 MeV/u CarbonNeutralNtuple on remote TOPAS host."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
HOST = "192.168.31.5"
USER = "v"
REMOTE_TOPAS = "/home/v/gpu/validation/topas"
LOCAL_TOPAS = Path("validation/topas")

UPLOAD = [
    "carbon_200MeVu_water.txt",
    "carbon_200MeVu_water_neutral_interactions.txt",
    "carbon_200MeVu_water_neutral_interactions_smoke.txt",
    "carbon_200MeVu_water_neutral_interactions_development_remote.txt",
    "run_neutral_remote.sh",
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode not in {"upload", "smoke", "development", "status", "fetch"}:
        print("Usage: _remote_neutral_e200.py [upload|smoke|development|status|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_neutral_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *neutral*|*CarbonNeutral*|*run_neutral*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/neutral* 2>/dev/null | head -40 || true
tail -n 12 /home/v/gpu/validation/topas/output/neutral-development_nohup.log 2>/dev/null || true
grep -E 'START|DONE|Total number of histories|Results have been written|Elapsed times|Total:' \
  /home/v/gpu/validation/topas/output/neutral-development_topas.log 2>/dev/null | tail -20 || true
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
            "neutral_smoke_interactions.header",
            "neutral_smoke_interactions.phsp",
            "neutral-smoke_topas.log",
            "neutral_development_interactions.header",
            "neutral_development_interactions.phsp",
            "neutral-development_topas.log",
            "neutral-development_nohup.log",
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
        print("running neutral smoke...", flush=True)
        _, stdout, stderr = client.exec_command(
            "cd ~/gpu && bash validation/topas/run_neutral_remote.sh smoke",
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
mkdir -p validation/topas/output
echo "[$(date -Is)] START neutral e200 development" | tee validation/topas/output/neutral-development_nohup.log
bash validation/topas/run_neutral_remote.sh development \
  >> validation/topas/output/neutral-development_nohup.log 2>&1
echo "[$(date -Is)] DONE neutral e200 development" | tee -a validation/topas/output/neutral-development_nohup.log
"""
    with sftp.file(f"{REMOTE_TOPAS}/output/run_neutral_development.sh", "w") as handle:
        handle.write(launcher)
    sftp.chmod(f"{REMOTE_TOPAS}/output/run_neutral_development.sh", 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && setsid bash validation/topas/output/run_neutral_development.sh "
        "</dev/null >/dev/null 2>&1 & sleep 4; "
        "pgrep -af 'opentopas-extension-install/bin/topas|run_neutral|neutral_development' || true; "
        "tail -n 8 validation/topas/output/neutral-development_nohup.log 2>/dev/null || true",
        timeout=30,
    )
    try:
        print(stdout.read().decode())
        print(stderr.read().decode())
    except Exception as exc:  # noqa: BLE001
        print("status read timed out (job may still be running):", exc)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
