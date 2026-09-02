#!/usr/bin/env python3
"""
tools/generate_step13_suite.py

Generates the complete test suite for Step 13 (Validate the C12 primary CT milestone):
1. Case 1: Homogeneous lung-like, soft-tissue, and dense-bone slabs (9 attenuation + 2 Bragg range).
2. Case 2: 5-material set adding air and trabecular bone (5 attenuation points).
3. Case 3: 25-section staircase phantom covering Section 0..24.
4. Case 4: Boundary-stress oblique entrance angle on 3D voxel grid.

Generates:
- CCTG v3 binary grids for MAIGO GPU simulation.
- TOPAS parameter files (.txt) matching the exact geometry and Schneider materials.
- SLURM submission scripts (.sh) conforming to cluster rules (<= 192 threads, <= 160 GB RAM).
- Full suite manifest (manifest.json).
"""

import json
import math
import os
import struct
from pathlib import Path

BASE_DIR = Path("/mnt/sda/wuwei/step13_primary_ct")
TOPAS_DIR = BASE_DIR / "topas"
SHARDS_DIR = TOPAS_DIR / "shards"
CCTG_DIR = BASE_DIR / "cctg"
GPU_DIR = BASE_DIR / "gpu"
EVIDENCE_DIR = BASE_DIR / "evidence"

REPO_ROOT = Path("/mnt/sdb/wuwei/MAIGO")
SCHNEIDER_TXT = REPO_ROOT / "data/HUtoMaterialSchneider.txt"
SCHNEIDER_XS_CSV = REPO_ROOT / "data/schneider/c12_schneider_inelastic_mass_xs.csv"
WATER_SP_CSV = REPO_ROOT / "data/stopping_power_water_geant4_11_3_2.csv"
STEP08_SHARDS = Path("/mnt/sda/wuwei/maigo-ct-schneider/step-08/shards")

CANONICAL_ZA_REL = [
    0.8992799520492554, 0.9917084574699402, 1.0029765367507935,
    1.0004115104675293, 0.9977754354476929, 0.9961254596710205,
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

CANONICAL_SECTION_PROPS = {
    0: {"name": "PatientTissueFromHUNegative975", "repHU": -975, "density": 0.0393235, "label": "air"},
    1: {"name": "PatientTissueFromHUNegative535", "repHU": -535, "density": 0.4700000, "label": "lung"},
    8: {"name": "PatientTissueFromHU100",          "repHU": 100,  "density": 1.0788000, "label": "soft_tissue"},
    12: {"name": "PatientTissueFromHU450",         "repHU": 450,  "density": 1.2966000, "label": "trabecular_bone"},
    20: {"name": "PatientTissueFromHU1250",        "repHU": 1250, "density": 1.8216000, "label": "dense_bone"},
}

def write_cctg_v3(filepath: Path, nx: int, ny: int, nz: int,
                    origin_x: float, origin_y: float, origin_z: float,
                    spacing_x: float, spacing_y: float, spacing_z: float,
                    densities: list, material_ids: list):
    assert len(densities) == nx * ny * nz
    assert len(material_ids) == nx * ny * nz
    magic = 0x47544343
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

TOPAS_PARAM_TEMPLATE = """includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = {beam_pos_x_mm:.4f} mm
d:Ge/BeamPosition/TransY = {beam_pos_y_mm:.4f} mm
d:Ge/BeamPosition/TransZ = {beam_pos_z_mm:.4f} mm
d:Ge/BeamPosition/RotX = {beam_rot_x_deg:.4f} deg
d:Ge/BeamPosition/RotY = {beam_rot_y_deg:.4f} deg
d:Ge/BeamPosition/RotZ = 0.0 deg

s:Ge/TargetBox/Parent = "World"
s:Ge/TargetBox/Type = "TsBox"
s:Ge/TargetBox/Material = "{material_name}"
d:Ge/TargetBox/HLX = {half_extent_x_mm:.4f} mm
d:Ge/TargetBox/HLY = {half_extent_y_mm:.4f} mm
d:Ge/TargetBox/HLZ = {half_thickness_mm:.4f} mm
d:Ge/TargetBox/TransX = 0.0 mm
d:Ge/TargetBox/TransY = 0.0 mm
d:Ge/TargetBox/TransZ = {trans_z_mm:.4f} mm

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
s:Sc/Validation/Component = "TargetBox"
s:Sc/Validation/OutputFile = "{scorer_base_out}"
s:Sc/Validation/OutputJsonPath = "{scorer_json_out}"
i:Sc/Validation/SectionId = {section_id}
s:Sc/Validation/MaterialName = "{material_name}"
u:Sc/Validation/NominalEnergyMeVPerU = {energy_mevu:.1f}
d:Sc/Validation/SlabThickness = {thickness_mm:.4f} mm
d:Sc/Validation/SlabTransZ = {trans_z_mm:.4f} mm
i:Sc/Validation/NumberOfDepthBins = {depth_bins}
s:Sc/Validation/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "TargetBox"
i:Sc/Dose3D/XBins = {dose_bins_x}
i:Sc/Dose3D/YBins = {dose_bins_y}
i:Sc/Dose3D/ZBins = {dose_bins_z}
s:Sc/Dose3D/OutputFile = "{dose_base_out}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = {threads}
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=s13_{id}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={mem_gb}G
#SBATCH --output=/mnt/sda/%u/step13_primary_ct/topas/job_{id}_%j.log
#SBATCH --error=/mnt/sda/%u/step13_primary_ct/topas/job_{id}_%j.err

echo "=== Starting SLURM Job $SLURM_JOB_ID for {id} at $(date -u) ==="
/home/wuwei/topas/topas-build/topas {param_file}
echo "=== Completed SLURM Job $SLURM_JOB_ID for {id} at $(date -u) ==="
"""

def generate_suite():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    TOPAS_DIR.mkdir(parents=True, exist_ok=True)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    CCTG_DIR.mkdir(parents=True, exist_ok=True)
    GPU_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    test_cases = []

    case1_points = [
        ("lung_100mevu", 1, 100.0, 10.0, 5, 100000, 7, 4, False, True),
        ("lung_200mevu", 1, 200.0, 25.0, 5, 100000, 14, 8, False, True),
        ("lung_300mevu", 1, 300.0, 40.0, 5, 100000, 21, 12, False, True),
        ("soft_tissue_100mevu", 8, 100.0, 4.0, 5, 100000, 7, 4, False, True),
        ("soft_tissue_200mevu", 8, 200.0, 10.0, 5, 100000, 14, 8, False, True),
        ("soft_tissue_300mevu", 8, 300.0, 20.0, 5, 100000, 21, 12, False, True),
        ("dense_bone_100mevu", 20, 100.0, 2.5, 5, 100000, 7, 4, False, True),
        ("dense_bone_200mevu", 20, 200.0, 6.0, 5, 100000, 14, 8, False, True),
        ("dense_bone_300mevu", 20, 300.0, 12.0, 5, 100000, 21, 12, False, True),
        ("soft_tissue_100mevu_bragg", 8, 100.0, 35.0, 35, 100000, 10, 8, True, False),
        ("dense_bone_100mevu_bragg", 20, 100.0, 25.0, 50, 100000, 10, 8, True, False),
    ]

    for pt in case1_points:
        cid, sec_id, e_mevu, thick, bins, hist, thr, mem, is_bragg, precomp = pt
        mat_info = CANONICAL_SECTION_PROPS[sec_id]
        test_cases.append({
            "id": cid,
            "category": "case1_homogeneous_slabs",
            "section_id": sec_id,
            "material_name": mat_info["name"],
            "material_label": mat_info["label"],
            "density": mat_info["density"],
            "energy_mevu": e_mevu,
            "thickness_mm": thick,
            "depth_bins": bins,
            "histories": hist,
            "threads": thr,
            "mem_gb": mem,
            "is_bragg_check": is_bragg,
            "precomputed_in_step08": precomp,
            "beam_angle_deg": 0.0,
        })

    case2_points = [
        ("air_100mevu", 0, 100.0, 50.0, 10, 100000, 10, 8),
        ("air_200mevu", 0, 200.0, 100.0, 10, 100000, 20, 16),
        ("trabecular_bone_100mevu", 12, 100.0, 3.0, 5, 100000, 10, 8),
        ("trabecular_bone_200mevu", 12, 200.0, 8.0, 5, 100000, 20, 16),
        ("trabecular_bone_300mevu", 12, 300.0, 15.0, 5, 100000, 30, 24),
    ]

    for pt in case2_points:
        cid, sec_id, e_mevu, thick, bins, hist, thr, mem = pt
        mat_info = CANONICAL_SECTION_PROPS[sec_id]
        test_cases.append({
            "id": cid,
            "category": "case2_five_material_set",
            "section_id": sec_id,
            "material_name": mat_info["name"],
            "material_label": mat_info["label"],
            "density": mat_info["density"],
            "energy_mevu": e_mevu,
            "thickness_mm": thick,
            "depth_bins": bins,
            "histories": hist,
            "threads": thr,
            "mem_gb": mem,
            "is_bragg_check": False,
            "precomputed_in_step08": False,
            "beam_angle_deg": 0.0,
        })

    test_cases.append({
        "id": "oblique_grid_200mevu",
        "category": "case4_boundary_stress",
        "section_id": 8,
        "material_name": CANONICAL_SECTION_PROPS[8]["name"],
        "material_label": "soft_tissue_oblique",
        "density": CANONICAL_SECTION_PROPS[8]["density"],
        "energy_mevu": 200.0,
        "thickness_mm": 20.0,
        "depth_bins": 10,
        "histories": 100000,
        "threads": 20,
        "mem_gb": 16,
        "is_bragg_check": False,
        "precomputed_in_step08": False,
        "beam_angle_deg": 15.0,
    })

    manifest_entries = []

    for tc in test_cases:
        cid = tc["id"]
        sec_id = tc["section_id"]
        thick = tc["thickness_mm"]
        bins = tc["depth_bins"]
        e_mevu = tc["energy_mevu"]
        hist = tc["histories"]
        thr = tc["threads"]
        mem = tc["mem_gb"]
        rot_y = tc["beam_angle_deg"]
        rot_x = 0.0

        nx, ny = 20, 20
        nz = bins
        spacing_x = 2.0
        spacing_y = 2.0
        spacing_z = thick / float(bins)
        origin_x = -0.5 * nx * spacing_x
        origin_y = -0.5 * ny * spacing_y
        origin_z = 0.0

        cctg_path = CCTG_DIR / f"{cid}.cctg"
        densities = [tc["density"]] * (nx * ny * nz)
        material_ids = [sec_id] * (nx * ny * nz)
        write_cctg_v3(cctg_path, nx, ny, nz, origin_x, origin_y, origin_z,
                      spacing_x, spacing_y, spacing_z, densities, material_ids)

        param_path = TOPAS_DIR / f"param_{cid}.txt"
        slurm_path = TOPAS_DIR / f"submit_{cid}.sh"
        json_out_path = SHARDS_DIR / f"validation_{cid}.json"
        dose_out_path = SHARDS_DIR / f"dose3d_{cid}"
        scorer_base_out = SHARDS_DIR / f"scorer_out_{cid}"

        if tc["precomputed_in_step08"]:
            src_json = STEP08_SHARDS / f"validation_{cid}.json"
            src_dose = STEP08_SHARDS / f"dose3d_{cid}.csv"
            dst_json = SHARDS_DIR / f"validation_{cid}.json"
            dst_dose = SHARDS_DIR / f"dose3d_{cid}.csv"
            if src_json.exists() and not dst_json.exists():
                os.symlink(src_json, dst_json)
            if src_dose.exists() and not dst_dose.exists():
                os.symlink(src_dose, dst_dose)

        half_thick = thick / 2.0
        trans_z = half_thick
        tot_energy = e_mevu * 12.0

        beam_z = -1.0
        param_content = TOPAS_PARAM_TEMPLATE.format(
            beam_pos_x_mm=0.0,
            beam_pos_y_mm=0.0,
            beam_pos_z_mm=beam_z,
            beam_rot_x_deg=rot_x,
            beam_rot_y_deg=rot_y,
            material_name=tc["material_name"],
            half_extent_x_mm=20.0,
            half_extent_y_mm=20.0,
            half_thickness_mm=half_thick,
            trans_z_mm=trans_z,
            total_energy_mev=tot_energy,
            energy_mevu=e_mevu,
            histories=hist,
            scorer_base_out=str(scorer_base_out),
            scorer_json_out=str(json_out_path),
            section_id=sec_id,
            thickness_mm=thick,
            depth_bins=bins,
            dose_bins_x=nx,
            dose_bins_y=ny,
            dose_bins_z=nz,
            dose_base_out=str(dose_out_path),
            threads=thr
        )
        param_path.write_text(param_content)

        slurm_content = SLURM_TEMPLATE.format(
            id=cid,
            threads=thr,
            mem_gb=mem,
            param_file=str(param_path)
        )
        slurm_path.write_text(slurm_content)
        slurm_path.chmod(0o755)

        req_gates = [3, 4, 5, 6, 7] if tc.get("is_bragg_check", False) else [1, 2, 4, 5, 6, 7]
        entry = {
            **tc,
            "required_gates": req_gates,
            "cctg_file": str(cctg_path),
            "param_file": str(param_path),
            "slurm_script": str(slurm_path),
            "topas_json_output": str(json_out_path),
            "topas_dose_csv": str(dose_out_path) + ".csv",
            "gpu_json_output": str(GPU_DIR / f"result_{cid}.json"),
            "nx": nx, "ny": ny, "nz": nz,
            "spacing_x_mm": spacing_x,
            "spacing_y_mm": spacing_y,
            "spacing_z_mm": spacing_z,
            "origin_x_mm": origin_x,
            "origin_y_mm": origin_y,
            "origin_z_mm": origin_z,
        }
        manifest_entries.append(entry)

    case3_id = "staircase_25sec_200mevu"
    staircase_nx, staircase_ny = 20, 20
    staircase_nz = 25
    staircase_spacing_x = 2.0
    staircase_spacing_y = 2.0
    staircase_spacing_z = 2.0
    staircase_origin_x = -0.5 * staircase_nx * staircase_spacing_x
    staircase_origin_y = -0.5 * staircase_ny * staircase_spacing_y
    staircase_origin_z = 0.0

    staircase_densities = []
    staircase_mat_ids = []

    with open("/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03/topas-schneider-materials.json") as f:
        topas_mats = json.load(f)["sections"]

    sec_densities = [s["density_g_cm3"] for s in topas_mats]

    for z in range(staircase_nz):
        for y in range(staircase_ny):
            for x in range(staircase_nx):
                staircase_densities.append(sec_densities[z])
                staircase_mat_ids.append(z)

    staircase_cctg = CCTG_DIR / f"{case3_id}.cctg"
    write_cctg_v3(staircase_cctg, staircase_nx, staircase_ny, staircase_nz,
                  staircase_origin_x, staircase_origin_y, staircase_origin_z,
                  staircase_spacing_x, staircase_spacing_y, staircase_spacing_z,
                  staircase_densities, staircase_mat_ids)

    staircase_param_path = TOPAS_DIR / f"param_{case3_id}.txt"
    staircase_slurm_path = TOPAS_DIR / f"submit_{case3_id}.sh"
    staircase_json_out = SHARDS_DIR / f"validation_{case3_id}.json"
    staircase_dose_out = SHARDS_DIR / f"dose3d_{case3_id}"
    staircase_scorer_base = SHARDS_DIR / f"scorer_out_{case3_id}"

    slices_def = ""
    for s_idx, s_info in enumerate(topas_mats):
        local_z = -24.0 + s_idx * 2.0
        slices_def += f"""
s:Ge/Slice_{s_idx}/Parent = "StaircaseBox"
s:Ge/Slice_{s_idx}/Type = "TsBox"
s:Ge/Slice_{s_idx}/Material = "{s_info['material_name']}"
d:Ge/Slice_{s_idx}/HLX = 20.0000 mm
d:Ge/Slice_{s_idx}/HLY = 20.0000 mm
d:Ge/Slice_{s_idx}/HLZ = 1.0000 mm
d:Ge/Slice_{s_idx}/TransZ = {local_z:.4f} mm
"""

    staircase_param_content = f"""includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/World/Invisible = "TRUE"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0000 mm
d:Ge/BeamPosition/TransY = 0.0000 mm
d:Ge/BeamPosition/TransZ = -1.0 mm
d:Ge/BeamPosition/RotX = 0.0000 deg
d:Ge/BeamPosition/RotY = 0.0000 deg
d:Ge/BeamPosition/RotZ = 0.0 deg

s:Ge/StaircaseBox/Parent = "World"
s:Ge/StaircaseBox/Type = "TsBox"
s:Ge/StaircaseBox/Material = "Vacuum"
d:Ge/StaircaseBox/HLX = 20.0000 mm
d:Ge/StaircaseBox/HLY = 20.0000 mm
d:Ge/StaircaseBox/HLZ = 25.0000 mm
d:Ge/StaircaseBox/TransZ = 25.0000 mm
{slices_def}

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "GenericIon(6,12)"
d:So/Beam/BeamEnergy = 2400.0 MeV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "None"
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 100000

s:Sc/Validation/Quantity = "CarbonSchneiderThinSlabValidationScorer"
s:Sc/Validation/Component = "StaircaseBox"
b:Sc/Validation/PropagateToChildren = "True"
s:Sc/Validation/OutputFile = "{staircase_scorer_base}"
s:Sc/Validation/OutputJsonPath = "{staircase_json_out}"
i:Sc/Validation/SectionId = 99
s:Sc/Validation/MaterialName = "SchneiderStaircase25"
u:Sc/Validation/NominalEnergyMeVPerU = 200.0
d:Sc/Validation/SlabThickness = 50.0000 mm
d:Sc/Validation/SlabTransZ = 25.0000 mm
i:Sc/Validation/NumberOfDepthBins = 25
s:Sc/Validation/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/Dose3D/Quantity = "DoseToMedium"
s:Sc/Dose3D/Component = "StaircaseBox"
b:Sc/Dose3D/PropagateToChildren = "True"
i:Sc/Dose3D/XBins = 20
i:Sc/Dose3D/YBins = 20
i:Sc/Dose3D/ZBins = 25
s:Sc/Dose3D/OutputFile = "{staircase_dose_out}"
s:Sc/Dose3D/OutputType = "csv"
s:Sc/Dose3D/IfOutputFileAlreadyExists = "Overwrite"

Ts/NumberOfThreads = 50
"""
    staircase_param_path.write_text(staircase_param_content)

    staircase_slurm_content = SLURM_TEMPLATE.format(
        id=case3_id,
        threads=50,
        mem_gb=40,
        param_file=str(staircase_param_path)
    )
    staircase_slurm_path.write_text(staircase_slurm_content)
    staircase_slurm_path.chmod(0o755)

    entry_case3 = {
        "id": case3_id,
        "category": "case3_staircase_25sec",
        "section_id": 999,
        "material_name": "SchneiderStaircase25",
        "material_label": "staircase_25sec",
        "density": 1.0,
        "energy_mevu": 200.0,
        "thickness_mm": 50.0,
        "depth_bins": 25,
        "histories": 100000,
        "threads": 50,
        "mem_gb": 40,
        "is_bragg_check": False,
        "precomputed_in_step08": False,
        "beam_angle_deg": 0.0,
        "required_gates": [1, 2, 4, 5, 6, 7],
        "cctg_file": str(staircase_cctg),
        "param_file": str(staircase_param_path),
        "slurm_script": str(staircase_slurm_path),
        "topas_json_output": str(staircase_json_out),
        "topas_dose_csv": str(staircase_dose_out) + ".csv",
        "gpu_json_output": str(GPU_DIR / f"result_{case3_id}.json"),
        "nx": staircase_nx, "ny": staircase_ny, "nz": staircase_nz,
        "spacing_x_mm": staircase_spacing_x,
        "spacing_y_mm": staircase_spacing_y,
        "spacing_z_mm": staircase_spacing_z,
        "origin_x_mm": staircase_origin_x,
        "origin_y_mm": staircase_origin_y,
        "origin_z_mm": staircase_origin_z,
    }
    manifest_entries.append(entry_case3)

    manifest_path = BASE_DIR / "manifest.json"
    manifest_data = {
        "schema_version": 1,
        "task": "Step 13 C12 Primary CT Schneider Validation Suite",
        "total_test_cases": len(manifest_entries),
        "slurm_new_jobs_threads": sum(e["threads"] for e in manifest_entries if not e.get("precomputed_in_step08", False) and e.get("slurm_script")),
        "slurm_new_jobs_mem_gb": sum(e["mem_gb"] for e in manifest_entries if not e.get("precomputed_in_step08", False) and e.get("slurm_script")),
        "cases": manifest_entries
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Successfully generated Step 13 suite with {len(manifest_entries)} test cases.")
    print(f"Manifest written to: {manifest_path}")
    print(f"SLURM new jobs: threads={manifest_data["slurm_new_jobs_threads"]} (<= 192), mem={manifest_data["slurm_new_jobs_mem_gb"]} GB (<= 160 GB)")

if __name__ == "__main__":
    generate_suite()
