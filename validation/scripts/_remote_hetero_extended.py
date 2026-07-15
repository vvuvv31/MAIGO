#!/usr/bin/env python3
"""Upload/run/fetch air-cavity and offset-bone TOPAS jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
HOST = "192.168.31.5"
USER = "v"
REMOTE = "/home/v/gpu/validation/topas"
LOCAL = Path("validation/topas")

CASES = {
    "air": {
        "upload": [
            "carbon_200MeVu_water_hetero_air_cavity.txt",
            "carbon_200MeVu_water_hetero_air_cavity_development_remote.txt",
        ],
        "param": "carbon_200MeVu_water_hetero_air_cavity_development_remote.txt",
        "log": "hetero-air-development_topas.log",
        "csv": "hetero_air_cavity_development_energy_deposit.csv",
    },
    "offset": {
        "upload": [
            "carbon_200MeVu_water_hetero_bone_offset.txt",
            "carbon_200MeVu_water_hetero_bone_offset_development_remote.txt",
        ],
        "param": "carbon_200MeVu_water_hetero_bone_offset_development_remote.txt",
        "log": "hetero-offset-development_topas.log",
        "csv": "hetero_bone_offset_development_energy_deposit.csv",
    },
}


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
    if len(sys.argv) < 3:
        print("Usage: _remote_hetero_extended.py [air|offset] [upload|development|status|fetch]")
        return 2
    case = sys.argv[1]
    mode = sys.argv[2]
    if case not in CASES or mode not in {"upload", "development", "status", "fetch"}:
        print("Usage: _remote_hetero_extended.py [air|offset] [upload|development|status|fetch]")
        return 2
    info = CASES[case]
    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in info["upload"]:
            sftp.put(str(LOCAL / name), f"{REMOTE}/{name}")
            print("uploaded", name)
        sftp.close()
        client.close()
        return 0
    if mode == "development":
        sftp = client.open_sftp()
        for name in info["upload"]:
            sftp.put(str(LOCAL / name), f"{REMOTE}/{name}")
            print("uploaded", name)
        launcher = f"""#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu/validation/topas
mkdir -p output
export TOPAS_G4_DATA_DIR="${{HOME}}/software/gate/G4DATA"
export LD_LIBRARY_PATH="${{HOME}}/software/gate/GATE/geant4-v11.1.3-install-MT/lib:${{HOME}}/gpu/build/opentopas-extension-install/lib:${{HOME}}/software/topas/gdcm-install/lib"
echo "[$(date -Is)] START hetero {case}"
"${{HOME}}/gpu/build/opentopas-extension-install/bin/topas" {info['param']} \\
  2>&1 | tee output/{info['log']}
echo "[$(date -Is)] DONE hetero {case}"
"""
        path = f"{REMOTE}/output/run_hetero_{case}_launcher.sh"
        with sftp.file(path, "w") as handle:
            handle.write(launcher)
        sftp.chmod(path, 0o755)
        sftp.close()
        _, stdout, _ = client.exec_command(
            f"cd ~/gpu && setsid bash validation/topas/output/run_hetero_{case}_launcher.sh "
            f"</dev/null >/dev/null 2>&1 & sleep 3; "
            f"pgrep -af 'opentopas-extension-install/bin/topas' | head -5; "
            f"tail -n 5 validation/topas/output/{info['log']} 2>/dev/null || true",
            timeout=30,
        )
        try:
            print(stdout.read().decode())
        except Exception as exc:  # noqa: BLE001
            print("status timed out:", exc)
        client.close()
        return 0
    if mode == "status":
        _, stdout, _ = client.exec_command(
            f"pgrep -af topas | head -8; "
            f"ls -la ~/gpu/validation/topas/output/{info['csv']} 2>/dev/null; "
            f"grep -E 'DONE|Total:|histories|Results' "
            f"~/gpu/validation/topas/output/{info['log']} 2>/dev/null | tail -12",
            timeout=40,
        )
        print(stdout.read().decode())
        client.close()
        return 0
    if mode == "fetch":
        out = Path("validation/topas/output")
        out.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        for name in [info["log"], info["csv"]]:
            try:
                sftp.get(f"{REMOTE}/output/{name}", str(out / name))
                print("fetched", name, (out / name).stat().st_size)
            except OSError as exc:
                print("missing", name, exc)
        sftp.close()
        client.close()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
