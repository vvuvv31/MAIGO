#!/usr/bin/env python3
"""Extract C-12 SP/XS tables for bone and lung on remote TOPAS."""

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
    "carbon_material_sp_bone_from_water.txt",
    "carbon_material_sp_lung_from_water.txt",
    "carbon_material_xs_bone_from_water.txt",
    "carbon_material_xs_lung_from_water.txt",
]

FETCH = [
    "carbon_c12_stopping_power_bone.phsp",
    "carbon_c12_stopping_power_bone.header",
    "carbon_c12_stopping_power_lung.phsp",
    "carbon_c12_stopping_power_lung.header",
    "carbon_c12_inelastic_cross_sections_bone.phsp",
    "carbon_c12_inelastic_cross_sections_bone.header",
    "carbon_c12_inelastic_cross_sections_lung.phsp",
    "carbon_c12_inelastic_cross_sections_lung.header",
    "material-sp-bone_topas.log",
    "material-sp-lung_topas.log",
    "material-xs-bone_topas.log",
    "material-xs-lung_topas.log",
]


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=20,
        allow_agent=False, look_for_keys=False,
    )
    return client


def main() -> int:
    if not PASSWORD:
        print("Set REMOTE_TOPAS_PASSWORD", file=sys.stderr)
        return 2
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"upload", "run", "fetch"}:
        print("Usage: _remote_material_tables.py [upload|run|fetch]")
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name, flush=True)
        sftp.close()
        client.close()
        return 0

    if mode == "fetch":
        out = Path("validation/topas/output")
        out.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        for name in FETCH:
            try:
                sftp.get(f"{REMOTE_TOPAS}/output/{name}", str(out / name))
                print("fetched", name, (out / name).stat().st_size, flush=True)
            except OSError as exc:
                print("missing", name, exc, flush=True)
        sftp.close()
        client.close()
        return 0

    sftp = client.open_sftp()
    for name in UPLOAD:
        sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
        print("uploaded", name, flush=True)
    sftp.close()

    cmd = r"""
set -euo pipefail
cd ~/gpu/validation/topas
mkdir -p output
export TOPAS_G4_DATA_DIR="${HOME}/software/gate/G4DATA"
export LD_LIBRARY_PATH="${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT/lib:${HOME}/gpu/build/opentopas-extension-install/lib:${HOME}/software/topas/gdcm-install/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
TOPAS="${HOME}/gpu/build/opentopas-extension-install/bin/topas"
run_one() {
  local param="$1"
  local log="$2"
  echo "[$(date -Is)] START $param"
  "$TOPAS" "$param" 2>&1 | tee "output/$log" | tail -n 12
  echo "[$(date -Is)] DONE $param"
}
run_one carbon_material_sp_bone_from_water.txt material-sp-bone_topas.log
run_one carbon_material_sp_lung_from_water.txt material-sp-lung_topas.log
run_one carbon_material_xs_bone_from_water.txt material-xs-bone_topas.log
run_one carbon_material_xs_lung_from_water.txt material-xs-lung_topas.log
ls -la output/carbon_c12_stopping_power_{bone,lung}.* output/carbon_c12_inelastic_cross_sections_{bone,lung}.*
for f in output/carbon_c12_stopping_power_{bone,lung}.header output/carbon_c12_inelastic_cross_sections_{bone,lung}.header; do
  echo "==== $f ===="; head -3 "$f"
done
"""
    print("running remote material table extraction...", flush=True)
    _, stdout, stderr = client.exec_command(cmd, timeout=1800, get_pty=True)
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="", flush=True)
    err = stderr.read().decode()
    if err:
        print(err, end="")
    code = stdout.channel.recv_exit_status()
    client.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
