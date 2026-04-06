#!/usr/bin/env bash
# Extract all .tar.gz under SRC (same layout as zip_dataset.sh output) into OUT.
# Usage: bash scripts/unzip_dataset.sh <download_dir> [output_dir]
#        Default output_dir is the same as download_dir.

set -euo pipefail

SRC=${1:?usage: $0 <download_dir> [output_dir]}
OUT=${2:-$SRC}

[[ -d "$SRC" ]] || { echo "Directory does not exist: $SRC"; exit 1; }
mkdir -p "$OUT"

echo "Extracting to: $OUT"
while IFS= read -r -d '' f; do
  echo "  $f"
  tar -xzf "$f" -C "$OUT"
done < <(find "$SRC" -type f -name '*.tar.gz' -print0)

echo "Done -> $OUT"
