#!/usr/bin/env python3
"""Run CarbonStoppingPowerNtuple on remote TOPAS and fetch outputs."""

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


def main() -> int:
    if not PASSWORD:
        print("Set REMOTE_TOPAS_PASSWORD", file=sys.stderr)
        return 2
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
    sftp = client.open_sftp()
    for name in (
        "carbon_200MeVu_water.txt",
        "carbon_200MeVu_water_stopping_power.txt",
    ):
        sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
        print("uploaded", name, flush=True)
    sftp.close()

    cmd = r"""
set -e
cd ~/gpu/validation/topas
mkdir -p output
export TOPAS_G4_DATA_DIR="${HOME}/software/gate/G4DATA"
export LD_LIBRARY_PATH="${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT/lib:${HOME}/gpu/build/opentopas-extension-install/lib:${HOME}/software/topas/gdcm-install/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${HOME}/gpu/build/opentopas-extension-install/bin/topas" carbon_200MeVu_water_stopping_power.txt \
  2>&1 | tee output/stopping-power_topas.log | tail -n 35
ls -la output/carbon_c12_stopping_power_water.*
grep -E 'Welcome to TOPAS|Geant4 version|Total:|unknown value' output/stopping-power_topas.log | head -10
"""
    print("running remote stopping-power extraction...", flush=True)
    _, stdout, stderr = client.exec_command(cmd, timeout=600, get_pty=True)
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="", flush=True)
    print(stderr.read().decode(), end="")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        client.close()
        return code

    out = Path("validation/topas/output")
    out.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    for name in (
        "carbon_c12_stopping_power_water.phsp",
        "carbon_c12_stopping_power_water.header",
        "stopping-power_topas.log",
    ):
        sftp.get(f"{REMOTE_TOPAS}/output/{name}", str(out / name))
        print("fetched", name, (out / name).stat().st_size, flush=True)
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
