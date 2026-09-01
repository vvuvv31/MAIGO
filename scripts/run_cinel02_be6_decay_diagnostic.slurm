#!/usr/bin/env bash
#SBATCH --job-name=cinbe6decay
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-be6-decay-diagnostic/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-be6-decay-diagnostic/logs/%x_%j.err

set -euo pipefail
readonly source_run=/mnt/sda/wuwei/cinel02-be6-secondary-smoke/z4a6_e200_h50000/run.txt
readonly root=/mnt/sda/wuwei/cinel02-be6-decay-diagnostic
readonly run_dir=${root}/z4a6_e200_h50000_v2
readonly topas_bin=/home/wuwei/topas/topas-build/topas
readonly seed=2026099609
readonly histories=50000

test -x "${topas_bin}"
test -s "${source_run}"
mkdir -p "${run_dir}" "${root}/logs"
{
    cat "${source_run}"
    cat <<'TOPAS_SCORER'

# Diagnostic-only vacuum box around the source position. TOPAS disallows
# scorers directly on World, while the Be-6 decay occurs before the phantom.
s:Ge/DecayRegion/Type = "TsBox"
s:Ge/DecayRegion/Parent = "World"
s:Ge/DecayRegion/Material = "G4_Galactic"
d:Ge/DecayRegion/HLX = 1.0 mm
d:Ge/DecayRegion/HLY = 1.0 mm
d:Ge/DecayRegion/HLZ = 1.0 mm
d:Ge/DecayRegion/TransX = 0.0 mm
d:Ge/DecayRegion/TransY = 0.0 mm
d:Ge/DecayRegion/TransZ = 2000.0 mm

# Diagnostic only: record every first-step daughter created by a radioactive
# decay in the source region. This scorer is not consumed by CINEL02 package tools.
s:Sc/Be6Decay/Quantity = "CarbonDecayNtuple"
s:Sc/Be6Decay/Component = "DecayRegion"
s:Sc/Be6Decay/OutputType = "ASCII"
s:Sc/Be6Decay/OutputFile = "be6_decay_daughters"
s:Sc/Be6Decay/IfOutputFileAlreadyExists = "Overwrite"
TOPAS_SCORER
} > "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|i:Ts/Seed = 2026099606|i:Ts/Seed = ${seed}|" "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|i:Ts/NumberOfThreads = 60|i:Ts/NumberOfThreads = ${SLURM_CPUS_PER_TASK}|" "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|i:So/PrimaryBeam/NumberOfHistoriesInRun = 50000|i:So/PrimaryBeam/NumberOfHistoriesInRun = ${histories}|" "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|d:Ge/World/HLZ = 2.0 m|d:Ge/World/HLZ = 3.0 m|" "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|d:Ge/DecayRegion/TransZ = 2000.0 mm|d:Ge/DecayRegion/TransZ = 2500.0 mm|" "${run_dir}/run_decay_diagnostic.txt"
sed -i "s|d:Ge/BeamPosition/TransZ = Ge/World/HLZ m|d:Ge/BeamPosition/TransZ = 2500.0 mm|" "${run_dir}/run_decay_diagnostic.txt"

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${run_dir}/raw
export CARBON_CINEL02_CAMPAIGN_UUID=00000000-0000-4000-8000-000000960212
export CARBON_CINEL02_RUN_TAG=z4a6_e200_h50000_decaydiagnostic
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=false
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02
export CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default
export CARBON_RANDOM_SEED_POLICY=${seed}

cd "${run_dir}"
mkdir -p "${CARBON_CINEL02_OUTPUT_DIR}"
time "${topas_bin}" run_decay_diagnostic.txt > topas.log 2>&1
test -s be6_decay_daughters.header
test -e be6_decay_daughters.phsp
