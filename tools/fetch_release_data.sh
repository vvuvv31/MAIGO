#!/usr/bin/env bash
# Download and unpack the pristine-v1 physics-data / CT-input release assets.
#
# The pristine branch intentionally ships only the small Git-tracked data
# tables. The large SHA-pinned physics packages and the RT07575 CT benchmark
# inputs live in the GitHub release so the source tree stays small.
#
#   ./tools/fetch_release_data.sh
#
# Environment overrides:
#   MAIGO_REPO      default vvuvv31/MAIGO
#   MAIGO_DATA_TAG  default pristine-v1
set -euo pipefail

REPO="${MAIGO_REPO:-vvuvv31/MAIGO}"
TAG="${MAIGO_DATA_TAG:-pristine-v1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Fetching ${TAG} assets from ${REPO} into ${ROOT}"
if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --dir "$TMP"
else
    base="https://github.com/${REPO}/releases/download/${TAG}"
    for asset in \
        pristine-data-em-v1.tar.gz \
        pristine-data-schneider-v1.tar.gz \
        pristine-ct-inputs-v1.tar.gz \
        SHA256SUMS; do
        curl -fL "${base}/${asset}" -o "${TMP}/${asset}"
    done
fi

if [ -f "${TMP}/SHA256SUMS" ]; then
    (cd "${TMP}" && sha256sum -c SHA256SUMS)
else
    echo "WARNING: SHA256SUMS missing; skipping integrity check" >&2
fi

for archive in "${TMP}"/pristine-*.tar.gz; do
    tar -xzf "${archive}" -C "${ROOT}"
done

echo "Data unpacked into ${ROOT}"
echo "CT inputs are under data/ct/ ; point ct_grid_file, tps_spots_file and"
echo "tps_beam_model_file at them (or run from this repository path)."
