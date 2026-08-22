#!/usr/bin/env bash
#SBATCH --job-name=proton_fluctuation_100k
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm_%x_%A_%a.out
#SBATCH --error=slurm_%x_%A_%a.err

set -euo pipefail
unset MODULEPATH || true

ROOT="${FLUCTUATION_ROOT:-/LustreData6/home/wuwei/maigo_fluctuation_proton_100k}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$ROOT/campaign}"
RUNNER="${RUNNER:-$ROOT/tools/proton_water/run_fluctuation_campaign.py}"
TOOLS_DIR="${TOOLS_DIR:-$ROOT/tools/package_tools}"
TOPAS_BIN="${TOPAS_BIN:-$ROOT/topas-install/bin/topas}"
G4_DATA_DIR="${G4_DATA_DIR:-/LustreData6/home/wuwei/Applications/GEANT4/G4DATA}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
WORKER_COUNT=5

SHARE=/ShareData1/opt/cascadelake/linux-centos7-cascadelake/gcc-9.4.0
PYTHON_BIN="${PYTHON_BIN:-${SHARE}/python-3.10.8-tzew3vzrsgbupmdbywnpblqeeygegvjc/bin/python3}"

if [[ ! "$WORKER_ID" =~ ^[0-4]$ ]]; then
    echo "invalid worker id: $WORKER_ID" >&2
    exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CAMPAIGN_DIR/campaign_manifest.json" ]] || {
    echo "missing campaign manifest: $CAMPAIGN_DIR/campaign_manifest.json" >&2
    exit 2
}

mapfile -t LABELS < <("$PYTHON_BIN" - "$CAMPAIGN_DIR" "$WORKER_ID" "$WORKER_COUNT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
worker = int(sys.argv[2])
workers = int(sys.argv[3])
campaign = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
points = campaign.get("points")
if not isinstance(points, list) or len(points) != campaign.get("grid", {}).get("point_count"):
    raise SystemExit("campaign point count is inconsistent")

selected = []
for ordinal, point in enumerate(points):
    if ordinal % workers != worker:
        continue
    manifest_path = (root / point["run_manifest"]["path"]).resolve()
    if not manifest_path.is_relative_to(root):
        raise SystemExit(f"run manifest escapes campaign root: {manifest_path}")
    run = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = (root / run["files"]["config"]["path"]).resolve()
    run_dir = config_path.parent
    prepared = run_dir / "fluctuation_point.json"
    result = run_dir / "run_result.json"
    if prepared.exists() != result.exists():
        raise SystemExit(f"incomplete prior output state for {point['label']}")
    if not result.exists():
        selected.append(point["label"])

for label in selected:
    print(label)
PY
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'dry-run worker=%s/%s partition=cpup6 nodes=1 cpus=56 selected=%s\n' \
        "$WORKER_ID" "$WORKER_COUNT" "${#LABELS[@]}"
    if (( ${#LABELS[@]} > 0 )); then
        printf 'first=%s\nlast=%s\n' "${LABELS[0]}" "${LABELS[-1]}"
    fi
    printf 'campaign=%s\nrunner=%s\ntopas=%s\n' \
        "$CAMPAIGN_DIR" "$RUNNER" "$TOPAS_BIN"
    exit 0
fi

[[ -f "$RUNNER" ]] || { echo "missing runner: $RUNNER" >&2; exit 2; }
[[ -f "$TOOLS_DIR/prepare_topas_fluctuation.py" ]] || {
    echo "missing fluctuation preparation tool: $TOOLS_DIR" >&2
    exit 2
}
[[ -f "$TOOLS_DIR/topas_ascii_ntuple.py" ]] || {
    echo "missing TOPAS ASCII parser: $TOOLS_DIR" >&2
    exit 2
}
[[ -x "$TOPAS_BIN" ]] || { echo "missing TOPAS executable: $TOPAS_BIN" >&2; exit 2; }
grep -a -q EnergyLossFluctuationNtuple "$TOPAS_BIN" || {
    echo "TOPAS lacks EnergyLossFluctuationNtuple: $TOPAS_BIN" >&2
    exit 2
}
[[ -d "$G4_DATA_DIR/G4EMLOW8.6.1" ]] || {
    echo "missing Geant4 11.3.2 data: $G4_DATA_DIR" >&2
    exit 2
}

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

if (( ${#LABELS[@]} == 0 )); then
    echo "worker=$WORKER_ID has no pending points"
    exit 0
fi

mkdir -p "$ROOT/status"
LABEL_FILE="$ROOT/status/worker_${WORKER_ID}_labels.txt"
printf '%s\n' "${LABELS[@]}" > "$LABEL_FILE"

COMMAND=(
    "$PYTHON_BIN" "$RUNNER" --run
    --campaign-dir "$CAMPAIGN_DIR"
    --topas-bin "$TOPAS_BIN"
    --g4-data-dir "$G4_DATA_DIR"
    --tools-dir "$TOOLS_DIR"
)
for label in "${LABELS[@]}"; do
    COMMAND+=(--point "$label")
done

start="$(date +%s)"
echo "[$(date -Is)] worker=$WORKER_ID host=$(hostname) points=${#LABELS[@]} start"
"${COMMAND[@]}"
wall_seconds="$(( $(date +%s) - start ))"

"$PYTHON_BIN" - "$CAMPAIGN_DIR" "$LABEL_FILE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
labels = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
campaign = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
points = {point["label"]: point for point in campaign["points"]}
for label in labels:
    manifest_path = root / points[label]["run_manifest"]["path"]
    run = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = (root / run["files"]["config"]["path"]).parent
    for name in ("fluctuation_point.json", "run_result.json"):
        if not (run_dir / name).is_file():
            raise SystemExit(f"missing completed output for {label}: {name}")
print(f"validated completed outputs for {len(labels)} points")
PY

printf '[%s] worker=%s points=%s wall_seconds=%s complete\n' \
    "$(date -Is)" "$WORKER_ID" "${#LABELS[@]}" "$wall_seconds"
