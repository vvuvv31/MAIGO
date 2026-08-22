#!/usr/bin/env bash
#SBATCH --job-name=proton_em_only_1M
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_%x_%A_%a.out
#SBATCH --error=slurm_%x_%A_%a.err

set -euo pipefail
unset MODULEPATH || true

ROOT="${EM_ONLY_ROOT:-/LustreData6/home/wuwei/maigo_fluctuation_proton_100k}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$ROOT/em_only_multienergy_1M}"
TOPAS_BIN="${TOPAS_BIN:-$ROOT/topas-install/bin/topas}"
G4_DATA_DIR="${G4_DATA_DIR:-/LustreData6/home/wuwei/Applications/GEANT4/G4DATA}"
TEMPLATE="${TEMPLATE:-$CAMPAIGN_DIR/proton_water_em_only_dose_1M.txt.in}"
ENERGIES=(70 100 150 200)

index="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [[ ! "$index" =~ ^[0-3]$ ]]; then
    echo "invalid array index: $index" >&2
    exit 2
fi
energy="${ENERGIES[$index]}"
seed="$((2026085000 + energy))"
run_dir="$CAMPAIGN_DIR/${energy}MeV"
config="$run_dir/run.txt"

[[ -x "$TOPAS_BIN" ]] || { echo "missing TOPAS: $TOPAS_BIN" >&2; exit 2; }
[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 2; }
[[ -d "$G4_DATA_DIR/G4EMLOW8.6.1" ]] || { echo "missing Geant4 data: $G4_DATA_DIR" >&2; exit 2; }

mkdir -p "$run_dir"
sed -e "s/@ENERGY@/$energy/g" -e "s/@SEED@/$seed/g" "$TEMPLATE" > "$config"

export TOPAS_G4_DATA_DIR="$G4_DATA_DIR"
export G4ENSDFSTATEDATA="$G4_DATA_DIR/G4ENSDFSTATE3.0"
export G4NEUTRONHPDATA="$G4_DATA_DIR/G4NDL4.7.1"
export G4LEDATA="$G4_DATA_DIR/G4EMLOW8.6.1"
export G4LEVELGAMMADATA="$G4_DATA_DIR/PhotonEvaporation6.1"
export G4RADIOACTIVEDATA="$G4_DATA_DIR/RadioactiveDecay6.1.2"
export G4PARTICLEXSDATA="$G4_DATA_DIR/G4PARTICLEXS4.1"
export G4SAIDXSDATA="$G4_DATA_DIR/G4SAIDDATA2.0"
export G4INCLDATA="$G4_DATA_DIR/G4INCL1.2"
export G4ABLADATA="$G4_DATA_DIR/G4ABLA3.3"
export G4PIIDATA="$G4_DATA_DIR/G4PII1.3"
export G4REALSURFACEDATA="$G4_DATA_DIR/RealSurface2.2"
export G4TENDL="$G4_DATA_DIR/G4TENDL1.4"
export LD_LIBRARY_PATH="$ROOT/topas-install/lib:/LustreData6/home/wuwei/Applications/GEANT4/geant4-install/lib64:/LustreData6/home/wuwei/Applications/TOPAS/OpenTOPAS-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

start="$(date +%s)"
echo "[$(date -Is)] energy=$energy host=$(hostname) start"
cd "$run_dir"
"$TOPAS_BIN" "$config" > topas.log 2>&1
output="proton_${energy}MeV_water_em_only_1M_total_dose.csv"
[[ -s "$output" ]] || { echo "missing output: $run_dir/$output" >&2; exit 2; }
printf '[%s] energy=%s wall_seconds=%s output=%s complete\n' \
    "$(date -Is)" "$energy" "$(( $(date +%s) - start ))" "$run_dir/$output"
