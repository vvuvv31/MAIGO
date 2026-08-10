#!/usr/bin/env bash
# One-click TOPAS material/database extraction.
#
# Edit the manual configuration block below before using --run. The default
# action is --dry-run so that an accidental invocation cannot start a large
# TOPAS job.

set -euo pipefail

###############################################################################
# Manual configuration: edit these paths and defaults for the target machine.
###############################################################################
TOPAS_BIN="/absolute/path/to/OpenTOPAS-install/bin/topas"
TOPAS_ENV_SCRIPT=""                 # optional: e.g. /absolute/path/to/topas-setup.sh
TOPAS_G4_DATA_DIR=""                # optional: directory containing G4DATA
TOPAS_LD_LIBRARY_PATH=""             # optional: colon-separated TOPAS/Geant4 libs
TOPAS_EXTENSIONS_DIR=""              # build-time path; normally startup/extensions

TOPAS_THREADS=40
HISTORIES_PER_ENERGY=20000
ENERGIES_MEV_U=(100 200 300 400)
PHANTOM_MATERIAL="G4_WATER"
PHANTOM_HALF_LENGTH_MM=350
SEED_BASE=20260810
ENABLE_SPECIES_TABLES=false

###############################################################################
# Paths. They can also be overridden by command-line options below.
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/templates/database_extraction.txt.in"
WORK_ROOT="${SCRIPT_DIR}/work"
OUTPUT_ROOT="${SCRIPT_DIR}/output"
MODE="dry-run"

usage() {
    cat <<'EOF'
Usage:
  startup/run_topas_database_extraction.sh [options]

Options:
  --dry-run                 Generate inputs and print commands (default).
  --run                     Run TOPAS sequentially for every selected energy.
  --energies LIST           Comma-separated MeV/u list, e.g. 100,200,400.
  --histories N             Histories per energy.
  --threads N               TOPAS worker threads.
  --material NAME           TOPAS material name, e.g. G4_WATER.
  --species-tables          Enable fragment/neutral/elastic lookup tables.
  --topas-bin PATH          Override TOPAS_BIN for this invocation.
  --env-script PATH         Override TOPAS_ENV_SCRIPT for this invocation.
  --g4-data PATH            Override TOPAS_G4_DATA_DIR for this invocation.
  --ld-library-path PATH    Override TOPAS_LD_LIBRARY_PATH for this invocation.
  --work-root PATH          Directory for generated inputs and raw TOPAS data.
  --output-root PATH        Directory for the run manifest.
  -h, --help                Show this help.

The TOPAS binary, environment script, Geant4 data path and library path are
configured at the top of this file. Use --run explicitly for real TOPAS work.
EOF
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --run)
            MODE="run"
            shift
            ;;
        --energies)
            [[ $# -ge 2 ]] || { echo "ERROR: --energies needs a value" >&2; exit 2; }
            IFS=',' read -r -a ENERGIES_MEV_U <<< "$2"
            shift 2
            ;;
        --histories)
            [[ $# -ge 2 ]] || { echo "ERROR: --histories needs a value" >&2; exit 2; }
            HISTORIES_PER_ENERGY="$2"
            shift 2
            ;;
        --threads)
            [[ $# -ge 2 ]] || { echo "ERROR: --threads needs a value" >&2; exit 2; }
            TOPAS_THREADS="$2"
            shift 2
            ;;
        --material)
            [[ $# -ge 2 ]] || { echo "ERROR: --material needs a value" >&2; exit 2; }
            PHANTOM_MATERIAL="$2"
            shift 2
            ;;
        --species-tables)
            ENABLE_SPECIES_TABLES=true
            shift
            ;;
        --topas-bin)
            [[ $# -ge 2 ]] || { echo "ERROR: --topas-bin needs a value" >&2; exit 2; }
            TOPAS_BIN="$2"
            shift 2
            ;;
        --env-script)
            [[ $# -ge 2 ]] || { echo "ERROR: --env-script needs a value" >&2; exit 2; }
            TOPAS_ENV_SCRIPT="$2"
            shift 2
            ;;
        --g4-data)
            [[ $# -ge 2 ]] || { echo "ERROR: --g4-data needs a value" >&2; exit 2; }
            TOPAS_G4_DATA_DIR="$2"
            shift 2
            ;;
        --ld-library-path)
            [[ $# -ge 2 ]] || { echo "ERROR: --ld-library-path needs a value" >&2; exit 2; }
            TOPAS_LD_LIBRARY_PATH="$2"
            shift 2
            ;;
        --work-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --work-root needs a value" >&2; exit 2; }
            WORK_ROOT="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --output-root needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "$TEMPLATE" ]] || die "missing TOPAS template: $TEMPLATE"
[[ -d "$SCRIPT_DIR/extensions" ]] || die "missing extension directory: $SCRIPT_DIR/extensions"

[[ "$TOPAS_THREADS" =~ ^[1-9][0-9]*$ ]] || die "TOPAS_THREADS must be a positive integer"
[[ "$HISTORIES_PER_ENERGY" =~ ^[1-9][0-9]*$ ]] || die "histories must be a positive integer"
[[ "$PHANTOM_HALF_LENGTH_MM" =~ ^[1-9][0-9]*$ ]] || die "phantom half length must be a positive integer"
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || die "SEED_BASE must be a non-negative integer"

for energy in "${ENERGIES_MEV_U[@]}"; do
    [[ "$energy" =~ ^[1-9][0-9]*$ ]] || die "invalid energy: $energy"
done

if [[ "$MODE" == "run" ]]; then
    [[ -n "$TOPAS_BIN" && "$TOPAS_BIN" != "/absolute/path/to/"* ]] || \
        die "edit TOPAS_BIN at the top of the script before using --run"

    if [[ -n "$TOPAS_ENV_SCRIPT" ]]; then
        [[ -f "$TOPAS_ENV_SCRIPT" ]] || die "TOPAS_ENV_SCRIPT does not exist: $TOPAS_ENV_SCRIPT"
        # shellcheck disable=SC1090
        source "$TOPAS_ENV_SCRIPT"
    fi

    if [[ -n "$TOPAS_G4_DATA_DIR" ]]; then
        [[ -d "$TOPAS_G4_DATA_DIR" ]] || die "TOPAS_G4_DATA_DIR does not exist: $TOPAS_G4_DATA_DIR"
        export TOPAS_G4_DATA_DIR
    fi
    if [[ -n "$TOPAS_LD_LIBRARY_PATH" ]]; then
        export LD_LIBRARY_PATH="${TOPAS_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    if [[ "$TOPAS_BIN" != */* ]]; then
        TOPAS_BIN="$(command -v "$TOPAS_BIN" || true)"
    fi
    [[ -n "$TOPAS_BIN" && -x "$TOPAS_BIN" ]] || die "TOPAS binary is not executable: $TOPAS_BIN"
    TOPAS_BIN="$(readlink -f "$TOPAS_BIN")"
fi

mkdir -p "$WORK_ROOT"
MANIFEST="${OUTPUT_ROOT}/run_manifest.tsv"
if [[ "$MODE" == "run" ]]; then
    mkdir -p "$OUTPUT_ROOT"
    if [[ ! -f "$MANIFEST" ]]; then
        printf 'energy_mev_u\tmaterial\thistories\tthreads\tspecies_tables\tstatus\tseconds\trun_dir\n' > "$MANIFEST"
    fi
fi

sed_escape() {
    printf '%s' "$1" | sed 's/[\\&|]/\\&/g'
}

render_config() {
    local energy="$1"
    local total_energy="$2"
    local seed="$3"
    local prefix="$4"
    local destination="$5"
    local material_escaped
    local prefix_escaped
    material_escaped="$(sed_escape "$PHANTOM_MATERIAL")"
    prefix_escaped="$(sed_escape "$prefix")"

    if [[ "$ENABLE_SPECIES_TABLES" == true ]]; then
        sed \
            -e 's/^#SPECIES# //' \
            -e "s|@ENERGY_MEV_U@|$(sed_escape "$energy")|g" \
            -e "s|@TOTAL_ENERGY_MEV@|$(sed_escape "$total_energy")|g" \
            -e "s|@SEED@|$(sed_escape "$seed")|g" \
            -e "s|@THREADS@|$(sed_escape "$TOPAS_THREADS")|g" \
            -e "s|@HISTORIES@|$(sed_escape "$HISTORIES_PER_ENERGY")|g" \
            -e "s|@MATERIAL@|${material_escaped}|g" \
            -e "s|@PHANTOM_HALF_LENGTH_MM@|$(sed_escape "$PHANTOM_HALF_LENGTH_MM")|g" \
            -e "s|@OUTPUT_PREFIX@|${prefix_escaped}|g" \
            "$TEMPLATE" > "$destination"
    else
        sed \
            -e '/^#SPECIES#/d' \
            -e "s|@ENERGY_MEV_U@|$(sed_escape "$energy")|g" \
            -e "s|@TOTAL_ENERGY_MEV@|$(sed_escape "$total_energy")|g" \
            -e "s|@SEED@|$(sed_escape "$seed")|g" \
            -e "s|@THREADS@|$(sed_escape "$TOPAS_THREADS")|g" \
            -e "s|@HISTORIES@|$(sed_escape "$HISTORIES_PER_ENERGY")|g" \
            -e "s|@MATERIAL@|${material_escaped}|g" \
            -e "s|@PHANTOM_HALF_LENGTH_MM@|$(sed_escape "$PHANTOM_HALF_LENGTH_MM")|g" \
            -e "s|@OUTPUT_PREFIX@|${prefix_escaped}|g" \
            "$TEMPLATE" > "$destination"
    fi
}

material_slug="${PHANTOM_MATERIAL//[^A-Za-z0-9_.-]/_}"
printf 'TOPAS database extraction: mode=%s material=%s histories/energy=%s threads=%s species_tables=%s\n' \
    "$MODE" "$PHANTOM_MATERIAL" "$HISTORIES_PER_ENERGY" "$TOPAS_THREADS" "$ENABLE_SPECIES_TABLES"
if [[ -n "$TOPAS_EXTENSIONS_DIR" ]]; then
    printf 'Configured build-time extension directory: %s\n' "$TOPAS_EXTENSIONS_DIR"
else
    printf 'Build-time extension directory: %s\n' "$SCRIPT_DIR/extensions"
fi

energy_index=0
for energy in "${ENERGIES_MEV_U[@]}"; do
    total_energy=$((12 * energy))
    seed=$((SEED_BASE + energy_index))
    run_dir="${WORK_ROOT}/e${energy}MeVu_${material_slug}"
    config_file="${run_dir}/database_extraction.txt"
    log_file="${run_dir}/topas.log"
    prefix="db_e${energy}MeVu_${material_slug}"
    mkdir -p "$run_dir"
    render_config "$energy" "$total_energy" "$seed" "$prefix" "$config_file"

    if [[ "$MODE" == "dry-run" ]]; then
        printf '[dry-run] energy=%s MeV/u histories=%s seed=%s\n' "$energy" "$HISTORIES_PER_ENERGY" "$seed"
        printf '          cd %q && %q %q > %q 2>&1\n' "$run_dir" "${TOPAS_BIN:-/path/to/topas}" "$config_file" "$log_file"
        ((energy_index += 1))
        continue
    fi

    printf '[run] energy=%s MeV/u histories=%s seed=%s\n' "$energy" "$HISTORIES_PER_ENERGY" "$seed"
    start_seconds="$(date +%s)"
    set +e
    (
        cd "$run_dir" || exit 1
        time -p "$TOPAS_BIN" "$(basename "$config_file")"
    ) > "$log_file" 2>&1
    status=$?
    set -e
    end_seconds="$(date +%s)"
    elapsed=$((end_seconds - start_seconds))

    if ((status != 0)); then
        printf '%s\t%s\t%s\t%s\t%s\tfailed(%s)\t%s\t%s\n' \
            "$energy" "$PHANTOM_MATERIAL" "$HISTORIES_PER_ENERGY" "$TOPAS_THREADS" \
            "$ENABLE_SPECIES_TABLES" "$status" "$elapsed" "$run_dir" >> "$MANIFEST"
        echo "TOPAS failed for ${energy} MeV/u; inspect ${log_file}" >&2
        tail -n 30 "$log_file" >&2 || true
        exit "$status"
    fi

    headers="$(find "$run_dir" -maxdepth 1 -type f -name '*.header' -printf '%f,' | sed 's/,$//')"
    if [[ -z "$headers" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\tno_headers\t%s\t%s\n' \
            "$energy" "$PHANTOM_MATERIAL" "$HISTORIES_PER_ENERGY" "$TOPAS_THREADS" \
            "$ENABLE_SPECIES_TABLES" "$elapsed" "$run_dir" >> "$MANIFEST"
        echo "TOPAS exited successfully but produced no .header files: ${run_dir}" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\tok\t%s\t%s\n' \
        "$energy" "$PHANTOM_MATERIAL" "$HISTORIES_PER_ENERGY" "$TOPAS_THREADS" \
        "$ENABLE_SPECIES_TABLES" "$elapsed" "$run_dir" >> "$MANIFEST"
    printf '      outputs: %s\n' "$headers"
    ((energy_index += 1))
done

if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry-run complete. Generated inputs are under: $WORK_ROOT"
else
    echo "TOPAS extraction complete. Manifest: $MANIFEST"
fi
