#!/usr/bin/env python3
"""Upload TOPAS extension sources and rebuild remote OpenTOPAS install."""

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

EXTENSIONS = [
    "CarbonCrossSectionNtuple.cc",
    "CarbonCrossSectionNtuple.hh",
    "CarbonStoppingPowerNtuple.cc",
    "CarbonStoppingPowerNtuple.hh",
    "CarbonCascadeNtuple.cc",
    "CarbonCascadeNtuple.hh",
    "CarbonNeutralNtuple.cc",
    "CarbonNeutralNtuple.hh",
    "CarbonReactionNtuple.cc",
    "CarbonReactionNtuple.hh",
    "CarbonDoseOrigin.cc",
    "CarbonDoseOrigin.hh",
]


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
    for name in EXTENSIONS:
        local = LOCAL_TOPAS / "extensions" / name
        remote = f"{REMOTE_TOPAS}/extensions/{name}"
        sftp.put(str(local), remote)
        print("uploaded", name, flush=True)
    sftp.put(
        str(LOCAL_TOPAS / "build_extensions_remote.sh"),
        f"{REMOTE_TOPAS}/build_extensions_remote.sh",
    )
    sftp.chmod(f"{REMOTE_TOPAS}/build_extensions_remote.sh", 0o755)
    sftp.close()

    print("rebuilding remote TOPAS extensions...", flush=True)
    _, stdout, stderr = client.exec_command(
        "cd ~/gpu && bash validation/topas/build_extensions_remote.sh",
        timeout=1800,
        get_pty=True,
    )
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
