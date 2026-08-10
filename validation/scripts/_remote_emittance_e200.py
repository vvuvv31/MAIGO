#!/usr/bin/env python3
"""Upload/run/fetch 200 MeV/u BiGaussian emittance TOPAS job."""

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
    "carbon_200MeVu_water_emittance.txt",
    "carbon_200MeVu_water_emittance_smoke.txt",
    "carbon_200MeVu_water_emittance_development_remote.txt",
    "run_emittance_remote.sh",
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
        print("Usage: _remote_emittance_e200.py [upload|smoke|development|status|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_emittance_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *emittance*|*run_emittance*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/emittance_200* 2>/dev/null || true
tail -n 12 /home/v/gpu/validation/topas/output/emittance-e200-development_nohup.log 2>/dev/null || true
grep -E 'START|DONE|Total number of histories|Results have been written|Elapsed times|Total:' \
  /home/v/gpu/validation/topas/output/emittance-e200-development_topas.log 2>/dev/null | tail -20 || true
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
            "emittance_200_development_dose_3d.csv",
            "emittance_200_development_idd.csv",
            "emittance-e200-development_topas.log",
            "emittance-e200-development_nohup.log",
            "emittance_200_smoke_dose_3d.csv",
            "emittance_200_smoke_idd.csv",
            "emittance-e200-smoke_topas.log",
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
        print("running emittance smoke...", flush=True)
        _, stdout, stderr = client.exec_command(
            "cd ~/gpu && bash validation/topas/run_emittance_remote.sh smoke",
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
echo "[$(date -Is)] START emittance e200 development" | tee validation/topas/output/emittance-e200-development_nohup.log
bash validation/topas/run_emittance_remote.sh development \
  >> validation/topas/output/emittance-e200-development_nohup.log 2>&1
echo "[$(date -Is)] DONE emittance e200 development" | tee -a validation/topas/output/emittance-e200-development_nohup.log
"""
    with sftp.file(f"{REMOTE_TOPAS}/output/run_emittance_e200_development.sh", "w") as handle:
        handle.write(launcher)
    sftp.chmod(f"{REMOTE_TOPAS}/output/run_emittance_e200_development.sh", 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && setsid bash validation/topas/output/run_emittance_e200_development.sh "
        "</dev/null >/dev/null 2>&1 & sleep 4; "
        "pgrep -af 'opentopas-extension-install/bin/topas|run_emittance' || true; "
        "tail -n 8 validation/topas/output/emittance-e200-development_nohup.log 2>/dev/null || true",
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
