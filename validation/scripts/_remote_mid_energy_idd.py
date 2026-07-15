#!/usr/bin/env python3
"""Upload and run 150/250/350 MeV/u total-IDD TOPAS jobs on remote host."""

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

ENERGIES = (150, 250, 350)

UPLOAD = [
    "run_mid_energy_idd_remote.sh",
]
for e in ENERGIES:
    UPLOAD.extend(
        [
            f"carbon_{e}MeVu_water.txt",
            f"carbon_{e}MeVu_water_smoke.txt",
            f"carbon_{e}MeVu_water_development_remote.txt",
        ]
    )
# base geometry/physics
UPLOAD.append("carbon_200MeVu_water.txt")


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
        print(
            "Usage: _remote_mid_energy_idd.py "
            "[upload|smoke|development|status|fetch]"
        )
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_mid_energy_idd_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *run_mid_energy*|*carbon_150MeVu*|*carbon_250MeVu*|*carbon_350MeVu*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/e15{0,0}_* /home/v/gpu/validation/topas/output/e250* /home/v/gpu/validation/topas/output/e350* 2>/dev/null | head -40 || true
ls -la /home/v/gpu/validation/topas/output/e150* /home/v/gpu/validation/topas/output/e250* /home/v/gpu/validation/topas/output/e350* 2>/dev/null | head -60 || true
tail -n 12 /home/v/gpu/validation/topas/output/mid-energy-development_nohup.log 2>/dev/null || true
for e in 150 250 350; do
  echo "=== e${e} ==="
  grep -E 'START|DONE|Total number of histories|Results have been written|Elapsed times|Total:' \
    /home/v/gpu/validation/topas/output/e${e}-development_topas.log 2>/dev/null | tail -8 || true
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
        names = []
        for e in ENERGIES:
            names.extend(
                [
                    f"e{e}_development_energy_deposit.csv",
                    f"e{e}_development_dose.csv",
                    f"e{e}_development_primary_c12_energy_deposit.csv",
                    f"e{e}-development_topas.log",
                ]
            )
        names.append("mid-energy-development_nohup.log")
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

    if mode == "smoke":
        print("running mid-energy smoke (150/250/350)...", flush=True)
        _, stdout, stderr = client.exec_command(
            "cd ~/gpu && bash validation/topas/run_mid_energy_idd_remote.sh all smoke",
            timeout=7200,
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

    # development: sequential 150 -> 250 -> 350, detached
    sftp = client.open_sftp()
    launcher = """#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu
mkdir -p validation/topas/output
log=validation/topas/output/mid-energy-development_nohup.log
echo "[$(date -Is)] START mid-energy 150/250/350 development" | tee "$log"
for e in 150 250 350; do
  echo "[$(date -Is)] START e${e}" | tee -a "$log"
  bash validation/topas/run_mid_energy_idd_remote.sh "$e" development \
    >> "$log" 2>&1
  echo "[$(date -Is)] DONE e${e}" | tee -a "$log"
done
echo "[$(date -Is)] DONE mid-energy development all" | tee -a "$log"
"""
    with sftp.file(f"{REMOTE_TOPAS}/output/run_mid_energy_development.sh", "w") as handle:
        handle.write(launcher)
    sftp.chmod(f"{REMOTE_TOPAS}/output/run_mid_energy_development.sh", 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && setsid bash validation/topas/output/run_mid_energy_development.sh "
        "</dev/null >/dev/null 2>&1 & sleep 3; "
        "for pid in /proc/[0-9]*; do "
        "cmd=$(tr '\\0' ' ' < \"$pid/cmdline\" 2>/dev/null || true); "
        "case \"$cmd\" in *run_mid_energy*|*carbon_150MeVu*|*carbon_250MeVu*|*carbon_350MeVu*|*opentopas-extension-install/bin/topas*) "
        "echo \"${pid##*/} $cmd\";; esac; done; "
        "tail -n 10 validation/topas/output/mid-energy-development_nohup.log 2>/dev/null || true",
        timeout=40,
    )
    print(stdout.read().decode())
    print(stderr.read().decode())
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
