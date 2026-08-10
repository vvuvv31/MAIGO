#!/usr/bin/env python3
"""Upload 1% energy-spread TOPAS params and launch remote multi-energy development jobs."""

from __future__ import annotations

import time
from pathlib import Path

import paramiko

HOST = "192.168.31.5"
USER = "v"
PASSWORD = "v"
REMOTE_ROOT = "/home/v/gpu"
REMOTE_TOPAS = f"{REMOTE_ROOT}/validation/topas"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=20)
    sftp = client.open_sftp()
    local_topas = Path("validation/topas")
    names = ["run_multi_energy_espread1_remote.sh"]
    for e in [100, 150, 200, 250, 300, 350, 400]:
        names.extend(
            [
                f"carbon_{e}MeVu_water_espread1.txt",
                f"carbon_{e}MeVu_water_espread1_development_remote.txt",
                f"carbon_{e}MeVu_water_espread1_smoke.txt",
            ]
        )
    for name in names:
        path = local_topas / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        sftp.put(str(path), f"{REMOTE_TOPAS}/{name}")
        print("put", name)

    launch = (
        f"mkdir -p {REMOTE_TOPAS}/output && "
        f"chmod +x {REMOTE_TOPAS}/run_multi_energy_espread1_remote.sh && "
        f"cd {REMOTE_ROOT} && "
        "nohup bash validation/topas/run_multi_energy_espread1_remote.sh all development "
        "> validation/topas/output/espread1_all_development.nohup.log 2>&1 </dev/null &"
    )
    # Use get_pty=False and do not wait on long job: fire-and-forget via bash -c
    transport = client.get_transport()
    assert transport is not None
    channel = transport.open_session()
    channel.exec_command(f"bash -lc {launch!r}")
    time.sleep(2)
    # Close without waiting for full TOPAS suite
    channel.close()

    stdin, stdout, stderr = client.exec_command(
        "pgrep -af topas | head -10; "
        "wc -l validation/topas/output/espread1_all_development.nohup.log 2>/dev/null; "
        "tail -n 15 /home/v/gpu/validation/topas/output/espread1_all_development.nohup.log 2>/dev/null",
        timeout=15,
    )
    print(stdout.read().decode(errors="replace"))
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print("stderr:", err)
    sftp.close()
    client.close()
    print("remote launch done")


if __name__ == "__main__":
    main()
