#!/usr/bin/env bash
set -euo pipefail
task_id=${1:?task id required}
species=("1 1" "1 2" "1 3" "2 3" "2 4" "2 6" "3 6" "3 7" "4 7" "4 9" "4 10" "5 8" "5 10" "5 11" "6 10" "6 11" "6 12")
energies=(10 25 50 100 200 400)
si=$((task_id / 6)); ei=$((task_id % 6)); read -r z a <<< "${species[si]}"; e=${energies[ei]}
case "${z}:${a}" in 1:1) particle=proton;; 1:2) particle=deuteron;; 1:3) particle=triton;; *) particle="GenericIon(${z},${a})";; esac
histories=50000; total=$((e*a)); seed=$((2026092100+task_id)); uuid=$(printf '00000000-0000-4000-8000-%012d' $((910000+task_id)))
tag=z${z}a${a}_e${e}_h${histories}_v3; root=/mnt/sda/wuwei/cinel02-hybrid-pilot/${tag}; raw=${root}/raw
template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
mkdir -p "${root}" "${raw}" /mnt/sda/wuwei/cinel02-hybrid-pilot/logs
sed -e "s|@SEED@|${seed}|g" -e "s|@THREADS@|${SLURM_CPUS_PER_TASK}|g" -e "s|@TOTAL_ENERGY@|${total}|g" \
 -e "s|@HISTORIES@|${histories}|g" -e 's|@ENERGY_BIN_MIN@|0.0|g' -e 's|@ENERGY_BIN_WIDTH@|1.0|g' \
 -e 's|@ENERGY_BIN_COUNT@|501|g' -e "s|GenericIon(6,12)|${particle}|g" \
 -e "s|ProjectileZ = 0|ProjectileZ = ${z}|g" -e "s|ProjectileA = 0|ProjectileA = ${a}|g" \
 -e 's|IncludeSecondaries = "TRUE"|IncludeSecondaries = "FALSE"|g' "${template}" > "${root}/run.txt"
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw} CARBON_CINEL02_CAMPAIGN_UUID=${uuid} CARBON_CINEL02_RUN_TAG=${tag}
export CARBON_CINEL02_PRIMARY_ONLY=true CARBON_CINEL02_OVERWRITE_CAMPAIGN=false CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02 CARBON_PHYSICS_LIST=FTFP_INCLXX CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default CARBON_RANDOM_SEED_POLICY=${seed}
cd "${root}"; time /home/wuwei/topas/topas-build/topas run.txt > topas.log 2>&1
grep -Fq "Total number of histories: ${histories}" topas.log
if find "${raw}/${uuid}" -name '*.cinel02' -size +63c -print -quit 2>/dev/null | grep -q .; then test -s "${raw}/${uuid}/cinel02.contract.json"; fi
