#!/bin/bash
#SBATCH --job-name=maigo-cu-slab
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail

repo=/mnt/sdb/wuwei/MAIGO_pristine
run_root=/mnt/sda/wuwei/minibeam_copper_slab_${SLURM_JOB_ID}
# First-pass matrix sized for the currently available node resources.  The
# omitted 200 MeV/u and intermediate 0.25/5/30 mm points are added only if
# these three path lengths expose a need for finer interpolation.
IFS=, read -r -a energies_mevu <<< "${SLAB_ENERGIES_CSV:-150,250,300}"
IFS=, read -r -a thicknesses_mm <<< "${SLAB_THICKNESSES_CSV:-1,10,60}"
# Process isolation: preserve all other physics and the incident seed.  With
# one energy, request 16 CPUs (each TOPAS instance below uses 16 threads).
disable_ion_elastic=${SLAB_DISABLE_ION_ELASTIC:-0}
if [[ "$disable_ion_elastic" != 0 && "$disable_ion_elastic" != 1 ]]; then
    echo "SLAB_DISABLE_ION_ELASTIC must be 0 or 1" >&2
    exit 1
fi
if ((16 * ${#energies_mevu[@]} > SLURM_CPUS_PER_TASK)); then
    echo "Request at least 16 CPUs per concurrent energy" >&2
    exit 1
fi

mkdir -p "${run_root}"
for thickness_mm in "${thicknesses_mm[@]}"; do
    half_mm=$(awk -v value="${thickness_mm}" 'BEGIN { printf "%.6f", value / 2.0 }')
    face_mm=$(awk -v value="${half_mm}" 'BEGIN { printf "%.6f", value + 0.015 }')
    pids=()
    for energy_mevu in "${energies_mevu[@]}"; do
        total_mev=$((energy_mevu * 12))
        case_dir="${run_root}/e${energy_mevu}_t${thickness_mm}mm"
        mkdir -p "${case_dir}/output"
        cp "${repo}/benchmark/carbonminibeam/run_copper_slab_phase_space.txt" \
            "${case_dir}/run.txt"
        if [[ "$disable_ion_elastic" == 1 ]]; then
            sed -i -e 's/Modules = 8 /Modules = 7 /' \
                -e 's/ "CarbonIonElasticPhysics"//' "${case_dir}/run.txt"
        fi
        sed -i \
            -e "s/i:Ts\/Seed = 26091801/i:Ts\/Seed = $((26091801 + energy_mevu)) /" \
            -e "s/d:Ge\/Copper\/HLZ = 30 mm/d:Ge\/Copper\/HLZ = ${half_mm} mm/" \
            -e "s/d:Ge\/BeamPosition\/TransZ = 30.015 mm/d:Ge\/BeamPosition\/TransZ = ${face_mm} mm/" \
            -e "s/d:Ge\/ExitFilm\/TransZ = -30.015 mm/d:Ge\/ExitFilm\/TransZ = -${face_mm} mm/" \
            -e "s/d:So\/CarbonBeam\/BeamEnergy = 3000 MeV/d:So\/CarbonBeam\/BeamEnergy = ${total_mev} MeV/" \
            "${case_dir}/run.txt"
        (
            cd "${case_dir}"
            export CARBON_CINEL02_OUTPUT_DIR="${case_dir}/raw"
            export CARBON_CINEL02_CAMPAIGN_UUID=20260918-0000-4000-8000-000000000001
            export CARBON_CINEL02_PRIMARY_ONLY=true
            export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
            export CARBON_CINEL02_RUN_TAG="cu_slab_e${energy_mevu}_t${thickness_mm}"
            export CARBON_TOPAS_VERSION=4.2.p3
            export CARBON_GEANT4_VERSION=11.3.2
            export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
            if [[ "$disable_ion_elastic" == 1 ]]; then
                export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,g4h-elastic_HP,g4stopping'
            fi
            export CARBON_PRODUCTION_CUTS='all=0.05mm'
            export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
            /home/wuwei/topas/topas-build/topas run.txt > run.log 2>&1
            ! grep -q 'Fatal Exception' run.log
            test -s output/slab_exit.phsp
        ) &
        pids+=("$!")
    done
    batch_failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            batch_failed=1
        fi
    done
    if ((batch_failed)); then
        echo "At least one TOPAS case failed for thickness ${thickness_mm} mm" >&2
        exit 1
    fi
done

printf '%s\n' "${run_root}" > "${run_root}/run_root.txt"
