#!/usr/bin/env python3
"""
tools/generate_step15_mcs_suite.py

Generates the Step 15 MCS validation benchmark suite:
- Narrow pencil beam simulations through 4 canonical materials + 2 interfaces
- 3D voxel dose scoring (40x40xNZ, voxel size 1 mm x 1 mm x 1 mm)
- CCTG v3 grid files, TOPAS parameter files, and Slurm batch scripts
- Dynamic Slurm allocation: total exactly 192 threads and 116 GB RAM
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path("/mnt/sdb/wuwei/MAIGO")
BASE_DIR = Path("/mnt/sda/wuwei/step15_schneider_mcs")
CCTG_DIR = BASE_DIR / "cctg"
TOPAS_DIR = BASE_DIR / "topas"
SHARDS_DIR = BASE_DIR / "shards"
GPU_DIR = BASE_DIR / "gpu"
MANIFEST_PATH = BASE_DIR / "manifest.json"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=mcs_{id}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={mem_gb}G
#SBATCH --output=/mnt/sda/%u/step15_schneider_mcs/job_{id}_%j.log
#SBATCH --error=/mnt/sda/%u/step15_schneider_mcs/job_{id}_%j.err

/home/wuwei/topas/topas-build/topas {param_file}
"""

CANONICAL_SECTION_PROPS = {
    1: {"name": "PatientTissueFromHUNegative535", "label": "lung", "density": 0.470041, "rad_len_g_cm2": 36.527496},
    8: {"name": "PatientTissueFromHU100", "label": "soft_tissue", "density": 1.078800, "rad_len_g_cm2": 37.288668},
    11: {"name": "PatientTissueFromHU350", "label": "trabecular_bone", "density": 1.231460, "rad_len_g_cm2": 34.171385},
    20: {"name": "PatientTissueFromHU1250", "label": "dense_bone", "density": 1.821600, "rad_len_g_cm2": 27.983677},
}

def write_cctg_v3(filepath, nx, ny, nz, dx, dy, dz, ox, oy, oz, densities, mat_ids):
    import struct
    with open(filepath, 'wb') as f:
        # Header: magic(4s), version(I), nx(I), ny(I), nz(I), ox(f), oy(f), oz(f), dx(f), dy(f), dz(f)
        header = struct.pack('<4sIIIIffffff', b'CCTG', 3, nx, ny, nz, ox, oy, oz, dx, dy, dz)
        f.write(header)
        # densities
        f.write(struct.pack(f'<{len(densities)}f', *densities))
        # material IDs
        f.write(struct.pack(f'<{len(mat_ids)}B', *mat_ids))
        # n_factors
        f.write(struct.pack('<I', 25))
        # za
        f.write(struct.pack('<25f', *([1.0] * 25)))
        # I
        f.write(struct.pack('<25f', *([75.0] * 25)))

def main():
    print("=== Step 15: Generating MCS Benchmark Suite ===")
    for d in [CCTG_DIR, TOPAS_DIR, SHARDS_DIR, GPU_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    cases_def = [
        {
            "id": "mcs_lung_150mevu",
            "type": "homogeneous",
            "section_id": 1,
            "energy_mevu": 150.0,
            "thickness_mm": 50.0,
            "nz": 50,
            "threads": 25,
            "mem_gb": 16,
            "histories": 100000
        },
        {
            "id": "mcs_soft_tissue_200mevu",
            "type": "homogeneous",
            "section_id": 8,
            "energy_mevu": 200.0,
            "thickness_mm": 40.0,
            "nz": 40,
            "threads": 35,
            "mem_gb": 20,
            "histories": 100000
        },
        {
            "id": "mcs_trabecular_bone_200mevu",
            "type": "homogeneous",
            "section_id": 11,
            "energy_mevu": 200.0,
            "thickness_mm": 30.0,
            "nz": 30,
            "threads": 35,
            "mem_gb": 20,
            "histories": 100000
        },
        {
            "id": "mcs_dense_bone_200mevu",
            "type": "homogeneous",
            "section_id": 20,
            "energy_mevu": 200.0,
            "thickness_mm": 25.0,
            "nz": 25,
            "threads": 30,
            "mem_gb": 20,
            "histories": 100000
        },
        {
            "id": "mcs_interface_tissue_bone_200mevu",
            "type": "interface",
            "layer1_sec": 8,
            "layer1_thick_mm": 20.0,
            "layer2_sec": 11,
            "layer2_thick_mm": 20.0,
            "energy_mevu": 200.0,
            "thickness_mm": 40.0,
            "nz": 40,
            "threads": 35,
            "mem_gb": 20,
            "histories": 100000
        },
        {
            "id": "mcs_interface_tissue_lung_200mevu",
            "type": "interface",
            "layer1_sec": 8,
            "layer1_thick_mm": 20.0,
            "layer2_sec": 1,
            "layer2_thick_mm": 30.0,
            "energy_mevu": 200.0,
            "thickness_mm": 50.0,
            "nz": 50,
            "threads": 32,
            "mem_gb": 20,
            "histories": 100000
        }
    ]

    manifest = {
        "schema_version": 1,
        "task": "Step 15 Multiple Coulomb Scattering Schneider Benchmark Matrix",
        "total_cases": len(cases_def),
        "total_threads_allocated": sum(c["threads"] for c in cases_def),
        "total_mem_allocated_gb": sum(c["mem_gb"] for c in cases_def),
        "cases": []
    }

    nx, ny = 40, 40
    dx, dy, dz = 1.0, 1.0, 1.0
    ox = -0.5 * nx * dx  # -20.0 mm
    oy = -0.5 * ny * dy  # -20.0 mm
    oz = 0.0

    for c in cases_def:
        cid = c["id"]
        nz = c["nz"]
        thick = c["thickness_mm"]
        e_mevu = c["energy_mevu"]
        beam_e_mev = e_mevu * 12.0
        hist = c["histories"]
        thr = c["threads"]
        mem = c["mem_gb"]

        cctg_file = CCTG_DIR / f"grid_{cid}.cctg"
        param_file = TOPAS_DIR / f"param_{cid}.txt"
        slurm_file = TOPAS_DIR / f"submit_{cid}.sh"
        dose_csv = SHARDS_DIR / f"dose3d_{cid}.csv"
        dose_out_base = SHARDS_DIR / f"dose3d_{cid}"
        gpu_json = GPU_DIR / f"result_{cid}.json"

        # Generate voxel materials & densities
        densities = []
        mat_ids = []

        if c["type"] == "homogeneous":
            sec_id = c["section_id"]
            rho = CANONICAL_SECTION_PROPS[sec_id]["density"]
            densities = [rho] * (nx * ny * nz)
            mat_ids = [sec_id] * (nx * ny * nz)
            mat_name = CANONICAL_SECTION_PROPS[sec_id]["name"]

            # TOPAS param content
            param_content = f"""includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"
# Pre-load all Schneider materials from synthetic phantom without geometry collision
s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"


s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = -100.0 mm
d:Ge/BeamPosition/RotX = 0.0 deg

s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Material = "{mat_name}"
d:Ge/Phantom/HLX = {0.5 * nx * dx:.4f} mm
d:Ge/Phantom/HLY = {0.5 * ny * dy:.4f} mm
d:Ge/Phantom/HLZ = {0.5 * nz * dz:.4f} mm
d:Ge/Phantom/TransX = 0.0 mm
d:Ge/Phantom/TransY = 0.0 mm
d:Ge/Phantom/TransZ = {0.5 * nz * dz:.4f} mm

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {beam_e_mev:.4f} MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {hist}

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "Phantom"
i:Sc/Dose3D/XBins = {nx}
i:Sc/Dose3D/YBins = {ny}
i:Sc/Dose3D/ZBins = {nz}
s:Sc/Dose3D/OutputFile = "{dose_out_base}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = {thr}
"""
        else:
            # Interface
            sec1 = c["layer1_sec"]
            thick1 = c["layer1_thick_mm"]
            sec2 = c["layer2_sec"]
            thick2 = c["layer2_thick_mm"]
            nz1 = int(round(thick1 / dz))
            nz2 = int(round(thick2 / dz))

            rho1 = CANONICAL_SECTION_PROPS[sec1]["density"]
            rho2 = CANONICAL_SECTION_PROPS[sec2]["density"]

            for z in range(nz):
                curr_sec = sec1 if z < nz1 else sec2
                curr_rho = rho1 if z < nz1 else rho2
                for _ in range(nx * ny):
                    densities.append(curr_rho)
                    mat_ids.append(curr_sec)

            mat1_name = CANONICAL_SECTION_PROPS[sec1]["name"]
            mat2_name = CANONICAL_SECTION_PROPS[sec2]["name"]

            # TOPAS param content with 2 adjacent slices in Phantom container
            param_content = f"""includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"
# Pre-load all Schneider materials from synthetic phantom without geometry collision
s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"


s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = -100.0 mm
d:Ge/BeamPosition/RotX = 0.0 deg

s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Material = "Vacuum"
d:Ge/Phantom/HLX = {0.5 * nx * dx:.4f} mm
d:Ge/Phantom/HLY = {0.5 * ny * dy:.4f} mm
d:Ge/Phantom/HLZ = {0.5 * nz * dz:.4f} mm
d:Ge/Phantom/TransX = 0.0 mm
d:Ge/Phantom/TransY = 0.0 mm
d:Ge/Phantom/TransZ = {0.5 * nz * dz:.4f} mm

s:Ge/Layer1/Parent = "Phantom"
s:Ge/Layer1/Type = "TsBox"
s:Ge/Layer1/Material = "{mat1_name}"
d:Ge/Layer1/HLX = {0.5 * nx * dx:.4f} mm
d:Ge/Layer1/HLY = {0.5 * ny * dy:.4f} mm
d:Ge/Layer1/HLZ = {0.5 * thick1:.4f} mm
d:Ge/Layer1/TransX = 0.0 mm
d:Ge/Layer1/TransY = 0.0 mm
d:Ge/Layer1/TransZ = {-0.5 * nz * dz + 0.5 * thick1:.4f} mm

s:Ge/Layer2/Parent = "Phantom"
s:Ge/Layer2/Type = "TsBox"
s:Ge/Layer2/Material = "{mat2_name}"
d:Ge/Layer2/HLX = {0.5 * nx * dx:.4f} mm
d:Ge/Layer2/HLY = {0.5 * ny * dy:.4f} mm
d:Ge/Layer2/HLZ = {0.5 * thick2:.4f} mm
d:Ge/Layer2/TransX = 0.0 mm
d:Ge/Layer2/TransY = 0.0 mm
d:Ge/Layer2/TransZ = {0.5 * nz * dz - 0.5 * thick2:.4f} mm

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {beam_e_mev:.4f} MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {hist}

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "Phantom"
b:Sc/Dose3D/PropagateToChildren = "True"
i:Sc/Dose3D/XBins = {nx}
i:Sc/Dose3D/YBins = {ny}
i:Sc/Dose3D/ZBins = {nz}
s:Sc/Dose3D/OutputFile = "{dose_out_base}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = {thr}
"""

        # Write CCTG binary
        write_cctg_v3(cctg_file, nx, ny, nz, dx, dy, dz, ox, oy, oz, densities, mat_ids)
        # Write TOPAS param file
        param_file.write_text(param_content)
        # Write Slurm script
        slurm_content = SLURM_TEMPLATE.format(
            id=cid,
            threads=thr,
            mem_gb=mem,
            param_file=str(param_file)
        )
        slurm_file.write_text(slurm_content)
        slurm_file.chmod(0o755)

        manifest["cases"].append({
            "id": cid,
            "type": c["type"],
            "section_id": c.get("section_id", c.get("layer1_sec")),
            "energy_mevu": e_mevu,
            "thickness_mm": thick,
            "histories": hist,
            "threads": thr,
            "mem_gb": mem,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "spacing_x_mm": dx,
            "spacing_y_mm": dy,
            "spacing_z_mm": dz,
            "origin_x_mm": ox,
            "origin_y_mm": oy,
            "origin_z_mm": oz,
            "cctg_file": str(cctg_file),
            "param_file": str(param_file),
            "slurm_script": str(slurm_file),
            "topas_dose_csv": str(dose_csv),
            "gpu_json_output": str(gpu_json)
        })

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(cases_def)} test cases in {MANIFEST_PATH}")
    print(f"Total threads allocated: {manifest['total_threads_allocated']} (max 192)")
    print(f"Total memory allocated: {manifest['total_mem_allocated_gb']} GB (max 160 GB)")

if __name__ == "__main__":
    main()
