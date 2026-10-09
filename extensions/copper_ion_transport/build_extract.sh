#!/usr/bin/env bash
set -euo pipefail
extension_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$extension_root/prepare.py" "$@"
