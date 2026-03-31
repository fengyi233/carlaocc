#!/bin/bash
# Generate all modalities for the occupancy dataset.
# Usage: bash occupancy_generation/gen_modalities.sh [hydra overrides...]
#   e.g. bash occupancy_generation/gen_modalities.sh town_names='["Town05_Opt"]' frame_range='[0,9]'

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "============================================================"
echo "  Step 1/3: Generating refined semantic depth maps"
echo "============================================================"
python "$SCRIPT_DIR/generators/gen_sem_depth.py" "$@"

echo "============================================================"
echo "  Step 2/3: Generating normal maps"
echo "============================================================"
python "$SCRIPT_DIR/generators/gen_normal.py" "$@"

echo "============================================================"
echo "  Step 3/3: Generating panoptic occupancy"
echo "============================================================"
python "$SCRIPT_DIR/generators/gen_pano_occ.py" "$@"

echo "============================================================"
echo "  All modalities generated successfully!"
echo "============================================================"
