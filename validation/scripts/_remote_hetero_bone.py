#!/usr/bin/env python3
"""Upload/run/fetch lateral bone-insert TOPAS jobs."""

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
    "carbon_200MeVu_water_hetero_bone_insert.txt",
    "carbon_200MeVu_water_hetero_bone_insert_smoke.txt",
    "carbon_200MeVu_water_hetero_bone_insert_development_remote.txt",
    "run_hetero_bone_remote.sh",
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode not in {"upload", "smoke", "development", "status", "fetch"}:
        print("Usage: _remote_hetero_bone.py [upload|smoke|development|status|fetch]")
        return 2
    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name)
        sftp.chmod(f"{REMOTE_TOPAS}/run_hetero_bone_remote.sh", 0o755)
        sftp.close()
        client.close()
        return 0
    if mode in {"smoke", "development"}:
        sftp = client.open_sftp()
        for name in UPLOAD:
            sftp.put(str(LOCAL_TOPAS / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name)
        sftp.chmod(f"{REMOTE_TOPAS}/run_hetero_bone_remote.sh", 0o755)
        launcher = f"""#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu
mkdir -p validation/topas/output
bash validation/topas/run_hetero_bone_remote.sh {mode} \\
  > validation/topas/output/hetero-bone-{mode}_nohup.log 2>&1
"""
        with sftp.file(
            f"{REMOTE_TOPAS}/output/run_hetero_bone_{mode}_launcher.sh", "w"
        ) as handle:
            handle.write(launcher)
        sftp.chmod(f"{REMOTE_TOPAS}/output/run_hetero_bone_{mode}_launcher.sh", 0o755)
        sftp.close()
        _, stdout, _ = client.exec_command(
            f"cd ~/gpu && setsid bash validation/topas/output/run_hetero_bone_{mode}_launcher.sh "
            f"</dev/null >/dev/null 2>&1 & sleep 4; "
            f"pgrep -af 'hetero|opentopas-extension-install/bin/topas' | head -8; "
            f"tail -n 8 validation/topas/output/hetero-bone-{mode}_nohup.log 2>/dev/null || true",
            timeout=30,
        )
        try:
            print(stdout.read().decode())
        except Exception as exc:  # noqa: BLE001
            print("status read timed out:", exc)
        client.close()
        return 0
    if mode == "status":
        _, stdout, _ = client.exec_command(
            "pgrep -af 'hetero|opentopas-extension-install/bin/topas' | head -10; "
            "ls -la ~/gpu/validation/topas/output/hetero_bone* 2>/dev/null | head; "
            "tail -n 15 ~/gpu/validation/topas/output/hetero-bone-development_nohup.log 2>/dev/null; "
            "grep -E 'DONE|Total:|Results have been written|histories' "
            "~/gpu/validation/topas/output/hetero-bone-development_topas.log 2>/dev/null | tail -12; "
            "grep -E 'DONE|Total:|Results have been written|histories' "
            "~/gpu/validation/topas/output/hetero-bone-smoke_topas.log 2>/dev/null | tail -8",
            timeout=40,
        )
        print(stdout.read().decode())
        client.close()
        return 0
    if mode == "fetch":
        out = Path("validation/topas/output")
        out.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        for name in [
            "hetero-bone-development_nohup.log",
            "hetero-bone-development_topas.log",
            "hetero-bone-smoke_nohup.log",
            "hetero-bone-smoke_topas.log",
            "hetero_bone_insert_development_energy_deposit.csv",
            "hetero_bone_insert_smoke_energy_deposit.csv",
        ]:
            try:
                sftp.get(f"{REMOTE_TOPAS}/output/{name}", str(out / name))
                print("fetched", name, (out / name).stat().st_size)
            except OSError as exc:
                print("missing", name, exc)
        sftp.close()
        client.close()
        return 0
    client.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
