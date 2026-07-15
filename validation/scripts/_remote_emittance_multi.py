#!/usr/bin/env python3
"""Upload/run/fetch multi-energy BiGaussian emittance TOPAS jobs."""

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
ENERGIES = (150, 200, 250, 350, 400)


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
    if mode not in {"upload", "development", "status", "fetch"}:
        print("Usage: _remote_emittance_multi.py [upload|development|status|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        names = [
            "carbon_200MeVu_water.txt",
            "carbon_emittance_base.txt",
            "run_emittance_multi_remote.sh",
        ]
        for e in ENERGIES:
            names.append(f"carbon_{e}MeVu_water_emittance.txt")
            names.append(f"carbon_{e}MeVu_water_emittance_development_remote.txt")
        for name in names:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_emittance_multi_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *emittance*|*run_emittance_multi*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/emittance_*_development_dose_3d.csv 2>/dev/null || true
tail -n 15 /home/v/gpu/validation/topas/output/emittance-multi-development_nohup.log 2>/dev/null || true
for e in 150 200 250 350 400; do
  echo "=== e${e} ==="
  grep -E 'DONE|Total:|Total number of histories|Results have been written' \
    /home/v/gpu/validation/topas/output/emittance-e${e}-development_topas.log 2>/dev/null | tail -6 || true
done
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
        names = ["emittance-multi-development_nohup.log"]
        for e in ENERGIES:
            names.extend(
                [
                    f"emittance_{e}_development_dose_3d.csv",
                    f"emittance_{e}_development_idd.csv",
                    f"emittance-e{e}-development_topas.log",
                ]
            )
        for name in names:
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

    # development detached sequential
    sftp = client.open_sftp()
    launcher = """#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu
mkdir -p validation/topas/output
log=validation/topas/output/emittance-multi-development_nohup.log
echo "[$(date -Is)] START emittance multi-energy development" | tee "$log"
for e in 150 200 250 350 400; do
  echo "[$(date -Is)] START e${e}" | tee -a "$log"
  bash validation/topas/run_emittance_multi_remote.sh "$e" >> "$log" 2>&1
  echo "[$(date -Is)] DONE e${e}" | tee -a "$log"
done
echo "[$(date -Is)] DONE emittance multi-energy all" | tee -a "$log"
"""
    with sftp.file(f"{REMOTE_TOPAS}/output/run_emittance_multi_development.sh", "w") as handle:
        handle.write(launcher)
    sftp.chmod(f"{REMOTE_TOPAS}/output/run_emittance_multi_development.sh", 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && setsid bash validation/topas/output/run_emittance_multi_development.sh "
        "</dev/null >/dev/null 2>&1 & sleep 4; "
        "pgrep -af 'opentopas-extension-install/bin/topas|run_emittance' || true; "
        "tail -n 10 validation/topas/output/emittance-multi-development_nohup.log 2>/dev/null || true",
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
