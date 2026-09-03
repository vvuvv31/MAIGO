#!/usr/bin/env python3
"""tools/generate_step21_level3.py

Generates Level 3 Synthetic Heterogeneous Phantom:
  lung -> soft tissue (with bone inclusion) -> bone (with air inclusion) -> soft tissue
Produces .cctg v3 grid, TOPAS reference simulation scripts, and SLURM launchers.
"""

import json
import math
import struct
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path("/mnt/sda/wuwei/step21_level3")
TOPAS_DIR = BASE_DIR / "topas"
TOPAS_DIR.mkdir(parents=True, exist_ok=True)
CCTG_DIR = REPO_DIR / "data" / "schneider"
CCTG_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_ZA_REL = [
    0.9995105266571045, 0.9984955787658691, 0.9964593648910522,
    0.9964593648910522, 0.9959663748741150, 0.9954733848571777,
    0.9943869709968567, 0.9917228817939758, 0.9835737347602844,
    0.9838401675224304, 0.9782578945159912, 0.9717891216278076,
    0.9662479758262634, 0.9616461992263794, 0.9570088982582092,
    0.9523995518684387, 0.9477784633636475, 0.9440864920616150,
    0.9412810206413269, 0.9375873804092407, 0.9348113536834717,
    0.9320344328880310, 0.9292566776275635, 0.9273962974548340,
    0.8279811739921570
]

CANONICAL_I_EV = [
    85.6829605102539,  69.6933822631836,  62.00094985961914,
    63.43705368041992, 64.97969818115234, 66.1916732788086,
    67.23116302490234, 69.33441925048828, 70.08238220214844,
    70.11997985839844, 72.77925872802734, 75.77842712402344,
    78.65174102783203, 81.14991760253906, 83.7420654296875,
    86.30131530761719, 89.0001220703125,  91.22447967529297,
    93.24040985107422, 95.57535552978516, 97.48521423339844,
    99.42965698242188, 101.4092025756836, 102.7972640991211,
    233.0
]

def write_cctg_v3(filepath: Path, nx: int, ny: int, nz: int,
                  origin_x: float, origin_y: float, origin_z: float,
                  spacing_x: float, spacing_y: float, spacing_z: float,
                  densities: list, material_ids: list):
    assert len(densities) == nx * ny * nz
    assert len(material_ids) == nx * ny * nz
    magic = 0x47544343  # CCTG
    version = 3

    with open(filepath, "wb") as f:
        f.write(struct.pack("<IIIII", magic, version, nx, ny, nz))
        f.write(struct.pack("<ffffff", origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z))
        f.write(struct.pack(f"<{len(densities)}f", *densities))
        f.write(struct.pack(f"<{len(material_ids)}B", *material_ids))
        n_factors = len(CANONICAL_ZA_REL)
        f.write(struct.pack("<I", n_factors))
        f.write(struct.pack(f"<{n_factors}f", *CANONICAL_ZA_REL))
        f.write(struct.pack(f"<{n_factors}f", *CANONICAL_I_EV))

def generate():
    nx, ny, nz = 40, 40, 75
    dx, dy, dz = 2.0, 2.0, 2.0
    ox, oy, oz = -40.0, -40.0, 0.0

    densities = []
    material_ids = []

    for iz in range(nz):
        z_mm = (iz + 0.5) * dz
        for iy in range(ny):
            y_mm = ox + (iy + 0.5) * dy
            for ix in range(nx):
                x_mm = oy + (ix + 0.5) * dx

                # Default layers:
                if z_mm < 30.0:
                    # Layer 1: Lung (HU = -535)
                    sec = 1
                    rho = 0.4800208
                elif z_mm < 70.0:
                    # Layer 2: Soft Tissue (HU = 100)
                    sec = 8
                    rho = 1.1199000
                    # Thin bone inclusion (HU = 1250)
                    if 48.0 <= z_mm <= 52.0 and -10.0 <= x_mm <= 10.0 and -10.0 <= y_mm <= 10.0:
                        sec = 20
                        rho = 1.7570000
                elif z_mm < 90.0:
                    # Layer 3: Bone (HU = 1250)
                    sec = 20
                    rho = 1.7570000
                    # Thin air inclusion (HU = -975)
                    if 78.0 <= z_mm <= 82.0 and -6.0 <= x_mm <= 6.0 and -6.0 <= y_mm <= 6.0:
                        sec = 0
                        rho = 0.0269525
                else:
                    # Layer 4: Soft Tissue (HU = 100)
                    sec = 8
                    rho = 1.1199000

                densities.append(rho)
                material_ids.append(sec)

    cctg_path = CCTG_DIR / "heterogeneous_level3.cctg"
    write_cctg_v3(cctg_path, nx, ny, nz, ox, oy, oz, dx, dy, dz, densities, material_ids)
    print(f"Generated {cctg_path} ({cctg_path.stat().st_size} bytes)")

    # Cases configuration
    cases = [
        {
            "id": "level3_axis_aligned",
            "energy_mevu": 220.0,
            "beam_pos_z": -50.0,
            "rot_x_deg": 0.0,
            "rot_y_deg": 0.0,
            "trans_x": 0.0,
            "trans_y": 0.0,
            "histories": 50000,
            "threads": 24
        },
        {
            "id": "level3_oblique",
            "energy_mevu": 260.0,
            "beam_pos_z": -50.0,
            "rot_x_deg": 15.0,
            "rot_y_deg": 0.0,
            "trans_x": 0.0,
            "trans_y": -10.0,
            "histories": 50000,
            "threads": 24
        }
    ]

    manifest = {
        "suite": "step21_level3",
        "cctg_path": str(cctg_path),
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "ox": ox, "oy": oy, "oz": oz,
        "cases": cases
    }
    (BASE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate TOPAS parameter files
    for c in cases:
        cid = c["id"]
        e_mevu = c["energy_mevu"]
        e_tot_mev = e_mevu * 12.0
        rot_x = c["rot_x_deg"]
        trans_y = c["trans_y"]
        hist = c["histories"]
        threads = c["threads"]

        txt_file = TOPAS_DIR / f"{cid}.txt"
        dose_out = TOPAS_DIR / f"{cid}_dose"
        json_out = TOPAS_DIR / f"{cid}_scorer.json"

        content = f"""includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

i:Ts/Seed = 20260921
i:Ts/NumberOfThreads = {threads}
b:Gr/Enable = "False"

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

# Beam Frame
s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = {c['trans_x']:.1f} mm
d:Ge/BeamPosition/TransY = {trans_y:.1f} mm
d:Ge/BeamPosition/TransZ = {c['beam_pos_z']:.1f} mm
d:Ge/BeamPosition/RotX = {rot_x:.1f} deg
d:Ge/BeamPosition/RotY = 0.0 deg
d:Ge/BeamPosition/RotZ = 0.0 deg

# Heterogeneous Phantom Parent Box
s:Ge/PhantomBox/Parent = "World"
s:Ge/PhantomBox/Type = "TsBox"
s:Ge/PhantomBox/Material = "Vacuum"
d:Ge/PhantomBox/HLX = 40.0 mm
d:Ge/PhantomBox/HLY = 40.0 mm
d:Ge/PhantomBox/HLZ = 75.0 mm
d:Ge/PhantomBox/TransX = 0.0 mm
d:Ge/PhantomBox/TransY = 0.0 mm
d:Ge/PhantomBox/TransZ = 75.0 mm

# Layer 1: Lung [0, 30] mm -> center at 15 mm -> relative to box center (75 mm) = -60 mm
s:Ge/Layer1_Lung/Parent = "PhantomBox"
s:Ge/Layer1_Lung/Type = "TsBox"
s:Ge/Layer1_Lung/Material = "PatientTissueFromHUNegative535"
d:Ge/Layer1_Lung/HLX = 40.0 mm
d:Ge/Layer1_Lung/HLY = 40.0 mm
d:Ge/Layer1_Lung/HLZ = 15.0 mm
d:Ge/Layer1_Lung/TransZ = -60.0 mm

# Layer 2: Soft Tissue [30, 70] mm -> center at 50 mm -> rel = -25 mm
s:Ge/Layer2_SoftTissue/Parent = "PhantomBox"
s:Ge/Layer2_SoftTissue/Type = "TsBox"
s:Ge/Layer2_SoftTissue/Material = "PatientTissueFromHU100"
d:Ge/Layer2_SoftTissue/HLX = 40.0 mm
d:Ge/Layer2_SoftTissue/HLY = 40.0 mm
d:Ge/Layer2_SoftTissue/HLZ = 20.0 mm
d:Ge/Layer2_SoftTissue/TransZ = -25.0 mm

# Bone inclusion in Layer 2: [48, 52] mm -> center at 50 mm -> rel inside Layer 2 = 0 mm
s:Ge/Inclusion_Bone/Parent = "Layer2_SoftTissue"
s:Ge/Inclusion_Bone/Type = "TsBox"
s:Ge/Inclusion_Bone/Material = "PatientTissueFromHU1250"
d:Ge/Inclusion_Bone/HLX = 10.0 mm
d:Ge/Inclusion_Bone/HLY = 10.0 mm
d:Ge/Inclusion_Bone/HLZ = 2.0 mm
d:Ge/Inclusion_Bone/TransZ = 0.0 mm

# Layer 3: Bone [70, 90] mm -> center at 80 mm -> rel = 5 mm
s:Ge/Layer3_Bone/Parent = "PhantomBox"
s:Ge/Layer3_Bone/Type = "TsBox"
s:Ge/Layer3_Bone/Material = "PatientTissueFromHU1250"
d:Ge/Layer3_Bone/HLX = 40.0 mm
d:Ge/Layer3_Bone/HLY = 40.0 mm
d:Ge/Layer3_Bone/HLZ = 10.0 mm
d:Ge/Layer3_Bone/TransZ = 5.0 mm

# Air cavity inside Layer 3: [78, 82] mm -> center at 80 mm -> rel inside Layer 3 = 0 mm
s:Ge/Inclusion_Air/Parent = "Layer3_Bone"
s:Ge/Inclusion_Air/Type = "TsBox"
s:Ge/Inclusion_Air/Material = "PatientTissueFromHUNegative975"
d:Ge/Inclusion_Air/HLX = 6.0 mm
d:Ge/Inclusion_Air/HLY = 6.0 mm
d:Ge/Inclusion_Air/HLZ = 2.0 mm
d:Ge/Inclusion_Air/TransZ = 0.0 mm

# Layer 4: Soft Tissue [90, 150] mm -> center at 120 mm -> rel = 45 mm
s:Ge/Layer4_SoftTissue/Parent = "PhantomBox"
s:Ge/Layer4_SoftTissue/Type = "TsBox"
s:Ge/Layer4_SoftTissue/Material = "PatientTissueFromHU100"
d:Ge/Layer4_SoftTissue/HLX = 40.0 mm
d:Ge/Layer4_SoftTissue/HLY = 40.0 mm
d:Ge/Layer4_SoftTissue/HLZ = 30.0 mm
d:Ge/Layer4_SoftTissue/TransZ = 45.0 mm

# Physics List
sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping"
d:Ph/Default/CutForAllParticles = 0.05 mm

# Beam Source
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {e_tot_mev:.2f} MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {hist}

# 3D Dose Scorer
s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "PhantomBox"
b:Sc/Dose3D/PropagateToChildren = "True"
i:Sc/Dose3D/XBins = 40
i:Sc/Dose3D/YBins = 40
i:Sc/Dose3D/ZBins = 75
s:Sc/Dose3D/OutputFile = "{dose_out}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"
"""
        txt_file.write_text(content, encoding="utf-8")

        slurm_file = TOPAS_DIR / f"run_{cid}.sh"
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=s21_{cid[:8]}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem=12G
#SBATCH --output={TOPAS_DIR}/job_{cid}_%j.log
#SBATCH --error={TOPAS_DIR}/job_{cid}_%j.err

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib:$LD_LIBRARY_PATH

/home/wuwei/topas/topas-build/topas {txt_file}
"""
        slurm_file.write_text(slurm_content, encoding="utf-8")
        slurm_file.chmod(0o755)
        print(f"Generated {txt_file} and {slurm_file}")

if __name__ == "__main__":
    generate()
