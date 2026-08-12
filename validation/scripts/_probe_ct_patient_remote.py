#!/usr/bin/env python3
import os
import paramiko

PASSWORD = os.environ.get("REMOTE_TOPAS_PASSWORD", "")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "192.168.31.5",
    username="v",
    password=PASSWORD,
    timeout=20,
    allow_agent=False,
    look_for_keys=False,
)
cmd = r"""
set -e
ls -la ~/gpu/validation/topas/ct_dicom | head
ls -la ~/gpu/validation/topas/output/run_ct_patient* 2>/dev/null || true
ls -la ~/gpu/validation/topas/output/ct* 2>/dev/null || true
ls -la ~/gpu/validation/topas/carbon_150MeVu_ct_patient*.txt ~/gpu/validation/topas/HUtoMaterialSchneider.txt 2>/dev/null || true
pgrep -af topas || echo no-topas
cd ~/gpu/validation/topas
export TOPAS_G4_DATA_DIR="${HOME}/software/geant4-v11.3.2-install/share/Geant4/data"
export LD_LIBRARY_PATH="${HOME}/software/geant4-v11.3.2-install/lib:${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon/lib:${HOME}/software/topas/gdcm-install/lib"
timeout 90 "${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon/bin/topas" carbon_150MeVu_ct_patient_smoke.txt 2>&1 | tail -60
"""
_, stdout, stderr = client.exec_command(cmd, timeout=120)
print(stdout.read().decode(errors="replace"))
err = stderr.read().decode(errors="replace")
if err:
    print("STDERR:", err[:3000])
client.close()
