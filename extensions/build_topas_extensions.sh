#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_topas_extensions.sh --topas-source DIR --build-dir DIR --profile PROFILE [options]

Profiles:
  nuclear-copper    CINEL03 capture/event writer, Copper tables and Schneider dumps
  unified-em        nuclear-copper plus the private Geant4 EM-table exporter
  all-ion-elastic   nuclear-copper plus current combined elastic/recoil exporter
  all-ion-elastic-production
                    exact source used for accepted all_ion_elastic_v1.bin
  elastic-recoil-production
                    exact source used for accepted elastic recoil stopping bank

The script creates a generated extension staging directory below BUILD_DIR and
never modifies the TOPAS source tree or the checked-in extension snapshots.

Options:
  --jobs N             Parallel build jobs (default: 8)
  --geant4-dir DIR     Directory containing Geant4Config.cmake
  --cmake-prefix PATH  CMake dependency prefix path (for example TOPAS/Qt)
  --topas-use-qt BOOL  Forward TOPAS_USE_QT (default: ON for the pinned stack)
  --cmake-arg ARG      Additional CMake configure argument; repeat as needed
EOF
}

topas_source=
build_dir=
profile=
jobs=8
geant4_dir=
cmake_prefix=
topas_use_qt=ON
cmake_args=()
while (($#)); do
  case "$1" in
    --topas-source) topas_source=$2; shift 2 ;;
    --build-dir) build_dir=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --geant4-dir) geant4_dir=$2; shift 2 ;;
    --cmake-prefix) cmake_prefix=$2; shift 2 ;;
    --topas-use-qt) topas_use_qt=$2; shift 2 ;;
    --cmake-arg) cmake_args+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n $topas_source && -n $build_dir && -n $profile ]] || { usage >&2; exit 2; }
[[ -f $topas_source/CMakeLists.txt ]] || { echo "Not a TOPAS source tree: $topas_source" >&2; exit 2; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=$build_dir/maigo-extension-source
mkdir -p "$stage"
find "$stage" -mindepth 1 -maxdepth 1 -type f -delete
cp "$here"/topas/common/* "$stage"/

case "$profile" in
  nuclear-copper) ;;
  unified-em)
    cp "$here/topas/unified_em/CarbonStoppingPowerNtuple.hh" "$stage/"
    # TOPAS discovers extension type from the first source line. The frozen
    # source intentionally lacks this marker because it was manually linked.
    { echo '// Scorer for CarbonStoppingPowerNtuple'; cat "$here/topas/unified_em/CarbonStoppingPowerNtuple.cc"; } \
      > "$stage/CarbonStoppingPowerNtuple.cc"
    ;;
  all-ion-elastic)
    cp "$here"/topas/all_ion_elastic/AllIonElasticDump.{cc,hh} "$stage"/
    ;;
  all-ion-elastic-production)
    cp "$here/topas/all_ion_elastic/AllIonElasticDump.hh" "$stage/"
    cp "$here/topas/all_ion_elastic/frozen_production/AllIonElasticDump.cc" \
      "$stage/AllIonElasticDump.cc"
    ;;
  elastic-recoil-production)
    cp "$here/topas/all_ion_elastic/AllIonElasticDump.hh" "$stage/"
    cp "$here/topas/all_ion_elastic/frozen_production/AllIonElasticDump_recoil.cc" \
      "$stage/AllIonElasticDump.cc"
    ;;
  *) echo "Unknown profile: $profile" >&2; usage >&2; exit 2 ;;
esac

configure_args=(-DTOPAS_EXTENSIONS_DIR="$stage" -DTOPAS_USE_QT="$topas_use_qt")
[[ -z $geant4_dir ]] || configure_args+=(-DGeant4_DIR="$geant4_dir")
[[ -z $cmake_prefix ]] || configure_args+=(-DCMAKE_PREFIX_PATH="$cmake_prefix")
configure_args+=("${cmake_args[@]}")

cmake -S "$topas_source" -B "$build_dir" "${configure_args[@]}"
cmake --build "$build_dir" --parallel "$jobs"
printf 'Built profile %s with extension source %s\n' "$profile" "$stage"
