#!/usr/bin/env python3
"""Upload/run/fetch density-slab TOPAS job on remote host."""

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
    "carbon_200MeVu_water_slab_density.txt",
    "carbon_200MeVu_water_slab_density_development_remote.txt",
    "carbon_200MeVu_water_slab_density_smoke.txt",
    "run_slab_density_remote.sh",
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
        print("Usage: _remote_slab_density.py [upload|smoke|development|status|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.chmod(f"{REMOTE_TOPAS}/run_slab_density_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            r"""
for pid in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *slab*|*run_slab_density*|*opentopas-extension-install/bin/topas*)
      echo "${pid##*/} $cmd" ;;
  esac
done
ls -la /home/v/gpu/validation/topas/output/slab_density_* 2>/dev/null || true
tail -n 20 /home/v/gpu/validation/topas/output/slab-density-development_nohup.log 2>/dev/null || true
grep -E 'DONE|Total:|Results have been written|ERROR|Exception' \
  /home/v/gpu/validation/topas/output/slab-density-development_topas.log 2>/dev/null | tail -20 || true
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
        names = [
            "slab-density-development_nohup.log",
            "slab-density-development_topas.log",
            "slab_density_development_layer1_energy_deposit.csv",
            "slab_density_development_dense_energy_deposit.csv",
            "slab_density_development_layer2_energy_deposit.csv",
            "slab_density_smoke_layer1_energy_deposit.csv",
            "slab_density_smoke_dense_energy_deposit.csv",
            "slab_density_smoke_layer2_energy_deposit.csv",
            "slab-density-smoke_topas.log",
        ]
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

    # smoke or development detached
    sftp = client.open_sftp()
    tag = mode
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu
mkdir -p validation/topas/output
log=validation/topas/output/slab-density-{tag}_nohup.log
echo "[$(date -Is)] START slab density {tag}" | tee "$log"
bash validation/topas/run_slab_density_remote.sh {tag} >> "$log" 2>&1
echo "[$(date -Is)] DONE slab density {tag}" | tee -a "$log"
"""
    remote_launcher = f"{REMOTE_TOPAS}/output/run_slab_density_{tag}_launcher.sh"
    with sftp.file(remote_launcher, "w") as handle:
        handle.write(launcher)
    sftp.chmod(remote_launcher, 0o755)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        f"cd ~/gpu && setsid bash validation/topas/output/run_slab_density_{tag}_launcher.sh "
        f"</dev/null >/dev/null 2>&1 & sleep 3; "
        f"pgrep -af 'opentopas-extension-install/bin/topas|run_slab_density' || true; "
        f"tail -n 8 validation/topas/output/slab-density-{tag}_nohup.log 2>/dev/null || true",
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
