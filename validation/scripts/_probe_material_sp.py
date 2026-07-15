#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
HOST = "192.168.31.5"
USER = "v"


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=20,
        allow_agent=False, look_for_keys=False,
    )
    sftp = client.open_sftp()
    for name in (
        "carbon_material_tables_base.txt",
        "carbon_material_sp_water_control.txt",
        "carbon_material_sp_bone.txt",
    ):
        sftp.put(f"validation/topas/{name}", f"/home/v/gpu/validation/topas/{name}")
        print("uploaded", name)
    sftp.close()
    cmd = r"""
set -e
cd ~/gpu/validation/topas
export TOPAS_G4_DATA_DIR="${HOME}/software/gate/G4DATA"
export LD_LIBRARY_PATH="${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT/lib:${HOME}/gpu/build/opentopas-extension-install/lib:${HOME}/software/topas/gdcm-install/lib"
TOPAS="${HOME}/gpu/build/opentopas-extension-install/bin/topas"
echo '=== WATER CONTROL ==='
"$TOPAS" carbon_material_sp_water_control.txt 2>&1 | tail -n 15
ls -la output/carbon_c12_stopping_power_water_control.*
head -8 output/carbon_c12_stopping_power_water_control.header
echo '=== BONE ==='
"$TOPAS" carbon_material_sp_bone.txt 2>&1 | tail -n 15
ls -la output/carbon_c12_stopping_power_bone.*
head -8 output/carbon_c12_stopping_power_bone.header
"""
    _, stdout, _ = client.exec_command(cmd, timeout=300, get_pty=True)
    print(stdout.read().decode())
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
