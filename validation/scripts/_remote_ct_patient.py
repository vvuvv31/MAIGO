#!/usr/bin/env python3
"""Upload DICOM + TOPAS patient CT job, run, fetch, optional compare vs GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
HOST = "192.168.31.5"
USER = "v"
REMOTE_TOPAS = "/home/v/gpu/validation/topas"
REMOTE_DICOM = f"{REMOTE_TOPAS}/ct_dicom"
LOCAL = Path("validation/topas")
LOCAL_DICOM = Path("ct/dicom")

PARAM_SMOKE = "carbon_150MeVu_ct_patient_smoke.txt"
PARAM_DEV = "carbon_150MeVu_ct_patient_development_remote.txt"
UPLOAD_FILES = [
    "carbon_150MeVu_ct_patient.txt",
    "carbon_150MeVu_ct_patient_smoke.txt",
    "carbon_150MeVu_ct_patient_development_remote.txt",
    "HUtoMaterialSchneider.txt",
]


def connect() -> paramiko.SSHClient:
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
    return client


def upload_dicom(sftp: paramiko.SFTPClient) -> None:
    try:
        sftp.mkdir(REMOTE_DICOM)
    except OSError:
        pass
    files = sorted(LOCAL_DICOM.glob("IMG*.dcm"))
    if not files:
        raise SystemExit(f"No IMG*.dcm under {LOCAL_DICOM}")
    for path in files:
        remote = f"{REMOTE_DICOM}/{path.name}"
        sftp.put(str(path), remote)
        print("uploaded dicom", path.name)
    # Skip RS structure set (not required for CT HU load).


def main() -> int:
    if not PASSWORD:
        print("Set REMOTE_TOPAS_PASSWORD", file=sys.stderr)
        return 2
    if len(sys.argv) < 2:
        print(
            "Usage: _remote_ct_patient.py "
            "[upload|smoke|development|status|fetch]"
        )
        return 2
    mode = sys.argv[1]
    if mode not in {"upload", "smoke", "development", "status", "fetch"}:
        print("Unknown mode", mode)
        return 2

    client = connect()
    if mode == "upload":
        sftp = client.open_sftp()
        for name in UPLOAD_FILES:
            sftp.put(str(LOCAL / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name)
        upload_dicom(sftp)
        sftp.close()
        client.close()
        return 0

    if mode in {"smoke", "development"}:
        sftp = client.open_sftp()
        for name in UPLOAD_FILES:
            sftp.put(str(LOCAL / name), f"{REMOTE_TOPAS}/{name}")
            print("uploaded", name)
        upload_dicom(sftp)
        param = PARAM_SMOKE if mode == "smoke" else PARAM_DEV
        log = (
            "ct-patient-smoke_topas.log"
            if mode == "smoke"
            else "ct-patient-development_topas.log"
        )
        launcher = f"""#!/usr/bin/env bash
set -euo pipefail
cd ~/gpu/validation/topas
mkdir -p output
export TOPAS_G4_DATA_DIR="${{HOME}}/software/geant4-v11.3.2-install/share/Geant4/data"
export LD_LIBRARY_PATH="${{HOME}}/software/geant4-v11.3.2-install/lib:${{HOME}}/software/topas/OpenTOPAS-install-v4.2.3-carbon/lib:${{HOME}}/software/topas/gdcm-install/lib"
echo "[$(date -Is)] START ct patient {mode}"
"${{HOME}}/software/topas/OpenTOPAS-install-v4.2.3-carbon/bin/topas" {param} \\
  2>&1 | tee output/{log}
echo "[$(date -Is)] DONE ct patient {mode}"
"""
        path = f"{REMOTE_TOPAS}/output/run_ct_patient_{mode}_launcher.sh"
        with sftp.file(path, "w") as handle:
            handle.write(launcher)
        sftp.chmod(path, 0o755)
        sftp.close()
        _, stdout, _ = client.exec_command(
            f"cd ~/gpu && setsid bash validation/topas/output/run_ct_patient_{mode}_launcher.sh "
            f"</dev/null >/dev/null 2>&1 & sleep 4; "
            f"pgrep -af 'opentopas-extension-install/bin/topas' | head -5; "
            f"tail -n 8 validation/topas/output/{log} 2>/dev/null || true",
            timeout=40,
        )
        try:
            print(stdout.read().decode())
        except Exception as exc:  # noqa: BLE001
            print("status timed out:", exc)
        client.close()
        return 0

    if mode == "status":
        _, stdout, _ = client.exec_command(
            "pgrep -af topas | head -10; "
            "ls -la ~/gpu/validation/topas/output/ct_patient* 2>/dev/null; "
            "grep -E 'DONE|Total:|histories|Results|error|Error|exiting' "
            "~/gpu/validation/topas/output/ct-patient-*.log 2>/dev/null | tail -20",
            timeout=40,
        )
        print(stdout.read().decode())
        client.close()
        return 0

    if mode == "fetch":
        out = Path("validation/topas/output")
        out.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        names = [
            "ct-patient-smoke_topas.log",
            "ct-patient-development_topas.log",
            "ct_patient_smoke_energy_deposit.csv",
            "ct_patient_development_energy_deposit.csv",
        ]
        for name in names:
            try:
                sftp.get(f"{REMOTE_TOPAS}/output/{name}", str(out / name))
                print("fetched", name, (out / name).stat().st_size)
            except OSError as exc:
                print("missing", name, exc)
        sftp.close()
        client.close()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
