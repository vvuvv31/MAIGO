#!/usr/bin/env python3
"""Poll remote espread1 TOPAS jobs; when done, fetch CSVs and postprocess."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import paramiko

HOST = "192.168.31.5"
USER = "v"
PASSWORD = "v"
REMOTE_OUT = "/home/v/gpu/validation/topas/output"
LOCAL_OUT = Path("validation/topas/output")
ENERGIES = [100, 150, 200, 250, 300, 350, 400]


def remote_status(client: paramiko.SSHClient) -> tuple[bool, str]:
    _, stdout, _ = client.exec_command(
        "ps -eo pid,cmd | grep -E 'espread1|opentopas-extension-install/bin' | "
        "grep -v grep || true; "
        "ls -1 /home/v/gpu/validation/topas/output/e*_espread1_development_energy_deposit.csv "
        "2>/dev/null | wc -l; "
        "tail -n 3 /home/v/gpu/validation/topas/output/espread1_all_development.nohup.log "
        "2>/dev/null || true"
    )
    text = stdout.read().decode(errors="replace")
    running = "opentopas" in text or "run_multi_energy_espread1" in text
    return running, text


def fetch_csvs(client: paramiko.SSHClient) -> list[Path]:
    sftp = client.open_sftp()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for e in ENERGIES:
        name = f"e{e}_espread1_development_energy_deposit.csv"
        rpath = f"{REMOTE_OUT}/{name}"
        lpath = LOCAL_OUT / name
        try:
            sftp.stat(rpath)
        except OSError:
            continue
        sftp.get(rpath, str(lpath))
        # also log if present
        log = f"e{e}-espread1-development_topas.log"
        try:
            sftp.get(f"{REMOTE_OUT}/{log}", str(LOCAL_OUT / log))
        except OSError:
            pass
        got.append(lpath)
        print("fetched", name, "bytes", lpath.stat().st_size)
    sftp.close()
    return got


def main() -> int:
    timeout_s = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
    poll_s = 30
    t0 = time.time()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=20)

    while True:
        running, status = remote_status(client)
        n_csv = 0
        for line in status.splitlines():
            if line.strip().isdigit():
                n_csv = int(line.strip())
        print(f"[{time.time()-t0:.0f}s] running={running} n_csv≈{n_csv}")
        print(status[-500:])
        if not running and n_csv >= len(ENERGIES):
            break
        if not running and (time.time() - t0) > 60 and n_csv > 0:
            # suite may have finished partially
            if n_csv >= len(ENERGIES):
                break
            # still wait a bit more if incomplete
            if (time.time() - t0) > timeout_s:
                break
        if time.time() - t0 > timeout_s:
            print("timeout")
            break
        time.sleep(poll_s)

    got = fetch_csvs(client)
    client.close()
    print(f"fetched {len(got)} CSVs")
    if len(got) < len(ENERGIES):
        print("incomplete; postprocess what we have")
    # postprocess local
    rc = subprocess.call(["cmd", "/c", "validation\\scripts\\postprocess_espread1_topas.cmd"])
    if rc != 0:
        return rc
    return subprocess.call(
        [sys.executable, "validation/scripts/compare_espread1_suite.py"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
