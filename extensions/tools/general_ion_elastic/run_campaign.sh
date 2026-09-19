#!/usr/bin/env bash
set -euo pipefail
if (($# != 2)); then
  echo "Usage: run_campaign.sh TOPAS_BINARY CAMPAIGN_DIR" >&2
  exit 2
fi
topas=$1
campaign=$(cd "$2" && pwd)
mkdir -p "$campaign/raw" "$campaign/logs"
while IFS= read -r case_name; do
  [[ -n $case_name ]] || continue
  stem=${case_name%.txt}
  "$topas" "$campaign/$case_name" > "$campaign/logs/$stem.log" 2>&1
done < "$campaign/cases.txt"
