#!/usr/bin/env python3
"""Kill any in-flight espread1 TOPAS jobs and restart a single sequential suite."""

from __future__ import annotations

import time

import paramiko

HOST = "192.168.31.5"
USER = "v"
PASSWORD = "v"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)

    list_cmd = (
        "ps -eo pid,cmd | "
        "grep -E 'espread1|opentopas-extension-install/bin' | "
        "grep -v grep || true"
    )
    _, stdout, _ = client.exec_command(list_cmd)
    before = stdout.read().decode()
    print("BEFORE:\n" + before)
    pids = []
    for line in before.splitlines():
        parts = line.strip().split(None, 1)
        if parts and parts[0].isdigit():
            pids.append(parts[0])
    if pids:
        kill_cmd = "kill -9 " + " ".join(pids)
        print("KILL:", kill_cmd)
        client.exec_command(kill_cmd)
        time.sleep(2)

    _, stdout, _ = client.exec_command(list_cmd)
    print("AFTER:\n" + stdout.read().decode())

    start = (
        "cd /home/v/gpu && "
        "nohup bash validation/topas/run_multi_energy_espread1_remote.sh all development "
        "> validation/topas/output/espread1_all_development.nohup.log 2>&1 </dev/null &"
    )
    client.exec_command(f"bash -lc {start!r}")
    time.sleep(3)
    _, stdout, _ = client.exec_command(list_cmd)
    print("RUNNING:\n" + stdout.read().decode())
    _, stdout, _ = client.exec_command(
        "tail -n 12 /home/v/gpu/validation/topas/output/espread1_all_development.nohup.log"
    )
    print("LOG:\n" + stdout.read().decode())
    client.close()


if __name__ == "__main__":
    main()
