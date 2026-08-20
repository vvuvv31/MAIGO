#!/bin/bash
#SBATCH --job-name=elastic_adaptive_b1
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH -n 56
#SBATCH --time=2-00:00:00
#SBATCH --array=0-15%5
#SBATCH --output=slurm_%x_%A_%a.out
#SBATCH --error=slurm_%x_%A_%a.err

set -euo pipefail
unset MODULEPATH || true
ROOT="${ADAPTIVE_ROOT:-/LustreData6/home/wuwei/maigo_elastic_adaptive_1000}"
MANIFEST="$ROOT/manifest.json"
RUNNER="$ROOT/tools/elastic_package/run_elastic_campaign.py"
TEMPLATE="$ROOT/tools/elastic_package/elastic_only.txt.in"
TOOLS="$ROOT/tools/package_tools"
TOPAS="$ROOT/topas-install/bin/topas"
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"

export TOPAS_G4_DATA_DIR=/LustreData6/home/wuwei/Applications/GEANT4/G4DATA
export G4ENSDFSTATEDATA="$TOPAS_G4_DATA_DIR/G4ENSDFSTATE3.0"
export G4NEUTRONHPDATA="$TOPAS_G4_DATA_DIR/G4NDL4.7.1"
export G4LEDATA="$TOPAS_G4_DATA_DIR/G4EMLOW8.6.1"
export G4LEVELGAMMADATA="$TOPAS_G4_DATA_DIR/PhotonEvaporation6.1"
export G4RADIOACTIVEDATA="$TOPAS_G4_DATA_DIR/RadioactiveDecay6.1.2"
export G4PARTICLEXSDATA="$TOPAS_G4_DATA_DIR/G4PARTICLEXS4.1"
export G4SAIDXSDATA="$TOPAS_G4_DATA_DIR/G4SAIDDATA2.0"
export G4INCLDATA="$TOPAS_G4_DATA_DIR/G4INCL1.2"
export G4ABLADATA="$TOPAS_G4_DATA_DIR/G4ABLA3.3"
export G4PIIDATA="$TOPAS_G4_DATA_DIR/G4PII1.3"
export G4REALSURFACEDATA="$TOPAS_G4_DATA_DIR/RealSurface2.2"
export G4TENDL="$TOPAS_G4_DATA_DIR/G4TENDL1.4"
export LD_LIBRARY_PATH="$ROOT/topas-install/lib:/LustreData6/home/wuwei/Applications/GEANT4/geant4-install/lib64:/LustreData6/home/wuwei/Applications/TOPAS/OpenTOPAS-install/lib:/LustreData6/home/wuwei/opengate_env/lib:/ShareData1/app/root6.28.02/lib:/LustreData6/home/wuwei/gate/geant4/lib64"

readarray -t FIELDS < <(python3 - "$MANIFEST" "$TASK_ID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
e = d["summary"]["first_batch_entries"][int(sys.argv[2])]
for k in ("bin", "representative_energy_MeV_per_u", "histories", "seed", "label", "event_probability", "deficit_events", "current_events"):
    print(e[k])
PY
)
BIN="${FIELDS[0]}"; ENERGY="${FIELDS[1]}"; HISTORIES="${FIELDS[2]}"; SEED="${FIELDS[3]}"
LABEL="${FIELDS[4]}"; PROBABILITY="${FIELDS[5]}"; DEFICIT="${FIELDS[6]}"; CURRENT="${FIELDS[7]}"
FINAL_DIR="$ROOT/runs/$LABEL"; TASK_ROOT="$ROOT/tasks/$TASK_ID"; STATUS="$ROOT/batch1_status/$TASK_ID.json"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "dry-run task=$TASK_ID bin=$BIN energy=$ENERGY histories=$HISTORIES seed=$SEED label=$LABEL partition=cpup6 nodes=1 n=56"
    echo "runner=$RUNNER topas=$TOPAS task_root=$TASK_ROOT final_dir=$FINAL_DIR"
    exit 0
fi

[[ -x "$TOPAS" ]] || { echo "missing TOPAS: $TOPAS" >&2; exit 2; }
grep -a -q CarbonElasticNtuple "$TOPAS" || { echo "TOPAS lacks CarbonElasticNtuple" >&2; exit 2; }
[[ -f "$RUNNER" && -f "$TEMPLATE" && -f "$TOOLS/prepare_topas_elastic.py" ]] || { echo "missing tools" >&2; exit 2; }
if [[ -e "$FINAL_DIR" || -e "$TASK_ROOT" ]]; then
    echo "refusing existing output/task root" >&2
    exit 3
fi
mkdir -p "$TASK_ROOT" "$(dirname "$STATUS")"
start="$(date +%s)"
python3 "$RUNNER" --run --projectile-z 1 --projectile-a 1 --material G4_WATER \
    --energies-mevu "$ENERGY" --histories "$HISTORIES" --threads 56 \
    --remote-root "$TASK_ROOT" --topas-bin "$TOPAS" --template "$TEMPLATE" \
    --tools-dir "$TOOLS" --physics-model G4HadronElasticPhysicsHP \
    --extraction-mode fixed-energy-package-sampling --seed-base "$SEED"
wall="$(( $(date +%s) - start ))"
TASK_DIR="$TASK_ROOT/runs/$LABEL"; META="$TASK_DIR/$LABEL.metadata.json"
python3 - "$META" "$HISTORIES" <<'PY'
import csv, gzip, hashlib, json, math, sys
from pathlib import Path
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["histories"] == int(sys.argv[2])
assert m["projectile"] == {"Z": 1, "A": 1} and m["material"] == "G4_WATER"
assert m["physics_model"] == "G4HadronElasticPhysicsHP"
p = m["provenance"]
assert p["processes"] == ["hadElastic"] and p["extraction_mode"] == "fixed-energy-package-sampling"
assert p["continuous_em_loss_enabled"] is False and p["not_for_dose_reference"] is True
for name in ("header", "phsp", "interactions", "products"):
    f = m["files"][name]; path = Path(f["path"])
    assert path.is_file() and path.stat().st_size > 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
for name in ("interactions", "products"):
    with gzip.open(m["files"][name]["path"], "rt", encoding="utf-8", newline="") as s:
        rows = list(csv.DictReader(s))
    assert len(rows) == m[name]
if m["entries"]:
    with gzip.open(m["files"]["interactions"]["path"], "rt", encoding="utf-8", newline="") as s:
        for row in csv.DictReader(s):
            assert row["process_name"] == "hadElastic"
            assert math.isfinite(float(row["incident_energy_MeV_per_u"]))
PY
mkdir -p "$ROOT/runs"; mv "$TASK_DIR" "$FINAL_DIR"
python3 - "$FINAL_DIR/$LABEL.metadata.json" "$STATUS" "$BIN" "$CURRENT" "$DEFICIT" "$PROBABILITY" "$SEED" "$wall" <<'PY'
import json, os, socket, sys
from pathlib import Path
m = json.load(open(sys.argv[1], encoding="utf-8")); actual = int(m["entries"]); current = int(sys.argv[4])
payload = {"status": "completed", "bin": int(sys.argv[3]), "current_events": current,
 "planned_deficit_events": int(sys.argv[5]), "actual_events": actual, "products": int(m["products"]),
 "planned_probability": float(sys.argv[6]), "yield_ratio": actual / m["histories"],
 "histories": m["histories"], "seed": int(sys.argv[7]), "wall_seconds": int(sys.argv[8]),
 "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
 "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "coverage_after": current + actual,
 "target_reached": current + actual >= 1000, "metadata": str(Path(sys.argv[1]).resolve())}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
echo "completed task=$TASK_ID bin=$BIN events=$(python3 -c "import json;print(json.load(open('$STATUS'))['actual_events'])") wall_seconds=$wall"
