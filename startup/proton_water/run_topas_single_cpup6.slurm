#!/usr/bin/env bash
#SBATCH --job-name=topas_single
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail
unset MODULEPATH || true

ROOT="${TOPAS_ROOT:-/LustreData6/home/wuwei/maigo_fluctuation_proton_100k}"
TOPAS_BIN="${TOPAS_BIN:-$ROOT/topas-install/bin/topas}"
G4_DATA_DIR="${G4_DATA_DIR:-/LustreData6/home/wuwei/Applications/GEANT4/G4DATA}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
CONFIG="${CONFIG:-run.txt}"
EXPECTED_OUTPUT="${EXPECTED_OUTPUT:?EXPECTED_OUTPUT is required}"

[[ -x "$TOPAS_BIN" ]] || { echo "missing TOPAS: $TOPAS_BIN" >&2; exit 2; }
[[ -f "$RUN_DIR/$CONFIG" ]] || { echo "missing config: $RUN_DIR/$CONFIG" >&2; exit 2; }

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

cd "$RUN_DIR"
"$TOPAS_BIN" "$CONFIG" > topas.log 2>&1
[[ -s "$EXPECTED_OUTPUT" ]] || { echo "missing output: $RUN_DIR/$EXPECTED_OUTPUT" >&2; exit 2; }
echo "completed: $RUN_DIR/$EXPECTED_OUTPUT"
