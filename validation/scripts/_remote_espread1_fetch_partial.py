#!/usr/bin/env python3
"""Fetch available espread1 TOPAS CSVs and compare with GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import paramiko

HOST = "192.168.31.5"
USER = "v"
PASSWORD = "v"
REMOTE_OUT = "/home/v/gpu/validation/topas/output"
LOCAL_OUT = Path("validation/topas/output")
ENERGIES = [100, 150, 200, 250, 300, 350, 400]


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    sftp = client.open_sftp()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    got = []
    for e in ENERGIES:
        name = f"e{e}_espread1_development_energy_deposit.csv"
        rpath = f"{REMOTE_OUT}/{name}"
        lpath = LOCAL_OUT / name
        try:
            sftp.stat(rpath)
        except OSError:
            continue
        sftp.get(rpath, str(lpath))
        print(f"fetched {name} ({lpath.stat().st_size} bytes)", flush=True)
        got.append(e)
    sftp.close()
    client.close()
    if not got:
        print("no CSVs yet")
        return 1
    # postprocess only available
    for e in got:
        raw = LOCAL_OUT / f"e{e}_espread1_development_energy_deposit.csv"
        out = Path(f"validation/results/espread1/topas_e{e}_idd.csv")
        meta = Path(f"validation/results/espread1/topas_e{e}.metadata.json")
        log = LOCAL_OUT / f"e{e}-espread1-development_topas.log"
        cmd = [
            sys.executable,
            "validation/scripts/prepare_topas_multi_energy_idd.py",
            "--energy-mevu",
            str(e),
            "--histories",
            "100000",
            "--raw-csv",
            str(raw),
            "--output-csv",
            str(out),
            "--metadata",
            str(meta),
            "--parameter-file",
            f"validation/topas/carbon_{e}MeVu_water_espread1_development_remote.txt",
        ]
        if log.exists():
            cmd.extend(["--log", str(log)])
        print("post", e, flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return subprocess.call(
        [
            sys.executable,
            "validation/scripts/compare_espread1_suite.py",
            "--energies",
            *[str(e) for e in got],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
