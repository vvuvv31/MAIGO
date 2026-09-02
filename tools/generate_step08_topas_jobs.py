#!/usr/bin/env python3
import json
from pathlib import Path

MATRIX = [
    # Lung-like: Section 1 (HU_rep = -535, density = 0.4700 g/cm3)
    {
        "id": "lung_100mevu",
        "material_label": "lung",
        "section_id": 1,
        "material_name": "PatientTissueFromHUNegative535",
        "energy_mevu": 100.0,
        "thickness_mm": 10.0,
        "bins": 5,
        "histories": 100000,
        "threads": 7,
        "mem_gb": 4
    },
    {
        "id": "lung_200mevu",
        "material_label": "lung",
        "section_id": 1,
        "material_name": "PatientTissueFromHUNegative535",
        "energy_mevu": 200.0,
        "thickness_mm": 25.0,
        "bins": 5,
        "histories": 100000,
        "threads": 14,
        "mem_gb": 8
    },
    {
        "id": "lung_300mevu",
        "material_label": "lung",
        "section_id": 1,
        "material_name": "PatientTissueFromHUNegative535",
        "energy_mevu": 300.0,
        "thickness_mm": 40.0,
        "bins": 5,
        "histories": 100000,
        "threads": 21,
        "mem_gb": 12
    },
    # Soft tissue: Section 8 (HU_rep = 100, density = 1.0788 g/cm3)
    {
        "id": "soft_tissue_100mevu",
        "material_label": "soft_tissue",
        "section_id": 8,
        "material_name": "PatientTissueFromHU100",
        "energy_mevu": 100.0,
        "thickness_mm": 4.0,
        "bins": 5,
        "histories": 100000,
        "threads": 7,
        "mem_gb": 4
    },
    {
        "id": "soft_tissue_200mevu",
        "material_label": "soft_tissue",
        "section_id": 8,
        "material_name": "PatientTissueFromHU100",
        "energy_mevu": 200.0,
        "thickness_mm": 10.0,
        "bins": 5,
        "histories": 100000,
        "threads": 14,
        "mem_gb": 8
    },
    {
        "id": "soft_tissue_300mevu",
        "material_label": "soft_tissue",
        "section_id": 8,
        "material_name": "PatientTissueFromHU100",
        "energy_mevu": 300.0,
        "thickness_mm": 20.0,
        "bins": 5,
        "histories": 100000,
        "threads": 21,
        "mem_gb": 12
    },
    # Dense bone: Section 20 (HU_rep = 1250, density = 1.8216 g/cm3)
    {
        "id": "dense_bone_100mevu",
        "material_label": "dense_bone",
        "section_id": 20,
        "material_name": "PatientTissueFromHU1250",
        "energy_mevu": 100.0,
        "thickness_mm": 2.5,
        "bins": 5,
        "histories": 100000,
        "threads": 7,
        "mem_gb": 4
    },
    {
        "id": "dense_bone_200mevu",
        "material_label": "dense_bone",
        "section_id": 20,
        "material_name": "PatientTissueFromHU1250",
        "energy_mevu": 200.0,
        "thickness_mm": 6.0,
        "bins": 5,
        "histories": 100000,
        "threads": 14,
        "mem_gb": 8
    },
    {
        "id": "dense_bone_300mevu",
        "material_label": "dense_bone",
        "section_id": 20,
        "material_name": "PatientTissueFromHU1250",
        "energy_mevu": 300.0,
        "thickness_mm": 12.0,
        "bins": 5,
        "histories": 100000,
        "threads": 21,
        "mem_gb": 12
    },
]

TOPAS_TEMPLATE = """includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

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

# Upstream beam source at Z = -1.0 m pointing +Z towards target
s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0 mm
d:Ge/BeamPosition/TransY = 0 mm
d:Ge/BeamPosition/TransZ = -1.0 m
d:Ge/BeamPosition/RotX = 0.0 deg

# Homogeneous 3D validation slab
s:Ge/Slab/Parent = "World"
s:Ge/Slab/Type = "TsBox"
s:Ge/Slab/Material = "{material_name}"
d:Ge/Slab/HLX = 5.0 cm
d:Ge/Slab/HLY = 5.0 cm
d:Ge/Slab/HLZ = {half_thickness_mm:.4f} mm
d:Ge/Slab/TransX = 0.0 mm
d:Ge/Slab/TransY = 0.0 mm
d:Ge/Slab/TransZ = 0.0 mm
i:Ge/Slab/NumberOfVoxelsX = 10
i:Ge/Slab/NumberOfVoxelsY = 10
i:Ge/Slab/NumberOfVoxelsZ = {bins}

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = {total_energy_mev:.1f} MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {histories}

s:Sc/Validation/Quantity = "CarbonSchneiderThinSlabValidationScorer"
s:Sc/Validation/Component = "Slab"
s:Sc/Validation/OutputFile = "{base_out_path}"
s:Sc/Validation/OutputJsonPath = "{json_out_path}"
i:Sc/Validation/SectionId = {section_id}
s:Sc/Validation/MaterialName = "{material_name}"
u:Sc/Validation/NominalEnergyMeVPerU = {energy_mevu:.1f}
d:Sc/Validation/SlabThickness = {thickness_mm:.4f} mm
d:Sc/Validation/SlabTransZ = 0.0 mm
i:Sc/Validation/NumberOfDepthBins = {bins}
s:Sc/Validation/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "Slab"
s:Sc/Dose3D/OutputFile = "{dose_out_path}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = {threads}
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=val_{id}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={mem_gb}G
#SBATCH --output=/mnt/sda/%u/maigo-ct-schneider/step-08/job_{id}_%j.log
#SBATCH --error=/mnt/sda/%u/maigo-ct-schneider/step-08/job_{id}_%j.err

echo "=== Starting SLURM Job $SLURM_JOB_ID for {id} at $(date -u) ==="
/home/wuwei/topas/topas-build/topas {param_file}
echo "=== Completed SLURM Job $SLURM_JOB_ID for {id} at $(date -u) ==="
"""

def main():
    base_dir = Path("/mnt/sda/wuwei/maigo-ct-schneider/step-08")
    shards_dir = base_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for pt in MATRIX:
        pt_id = pt["id"]
        param_path = base_dir / f"param_{pt_id}.txt"
        slurm_path = base_dir / f"submit_{pt_id}.sh"
        json_path = shards_dir / f"validation_{pt_id}.json"
        dose_path = shards_dir / f"dose3d_{pt_id}"
        base_out = shards_dir / f"scorer_out_{pt_id}"

        half_thick = pt["thickness_mm"] / 2.0
        tot_energy = pt["energy_mevu"] * 12.0

        param_content = TOPAS_TEMPLATE.format(
            material_name=pt["material_name"],
            half_thickness_mm=half_thick,
            thickness_mm=pt["thickness_mm"],
            bins=pt["bins"],
            total_energy_mev=tot_energy,
            energy_mevu=pt["energy_mevu"],
            histories=pt["histories"],
            section_id=pt["section_id"],
            base_out_path=str(base_out),
            json_out_path=str(json_path),
            dose_out_path=str(dose_path),
            threads=pt["threads"]
        )
        param_path.write_text(param_content)

        slurm_content = SLURM_TEMPLATE.format(
            id=pt_id,
            threads=pt["threads"],
            mem_gb=pt["mem_gb"],
            param_file=str(param_path)
        )
        slurm_path.write_text(slurm_content)
        slurm_path.chmod(0o755)

        manifest_entries.append({
            **pt,
            "param_file": str(param_path),
            "slurm_script": str(slurm_path),
            "json_output": str(json_path),
            "dose_csv_output": str(dose_path) + ".csv"
        })

    manifest_path = base_dir / "matrix_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "task": "Step 08 TOPAS Thin-Slab Validation Matrix",
        "total_points": len(manifest_entries),
        "total_threads_allocated": sum(e["threads"] for e in manifest_entries),
        "total_mem_allocated_gb": sum(e["mem_gb"] for e in manifest_entries),
        "points": manifest_entries
    }, indent=2))

    print(f"Generated {len(manifest_entries)} matrix point configs and SLURM scripts.")
    print(f"Total threads: {sum(e['threads'] for e in manifest_entries)} (limit 192)")
    print(f"Total memory: {sum(e['mem_gb'] for e in manifest_entries)} GB (limit 160 GB)")

if __name__ == "__main__":
    main()
