#!/bin/bash
# Visualize all modalities for the occupancy dataset.
# Usage: bash occupancy_generation/vis_modalities.sh [hydra overrides...]
#   e.g. bash occupancy_generation/vis_modalities.sh town_names='["Town05_Opt"]' frame_range='[0,9]' vis_interval=5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "============================================================"
echo "  Step 1/2: Visualizing meshes"
echo "============================================================"
python "$SCRIPT_DIR/visualizers/vis_mesh.py" "$@"

echo "============================================================"
echo "  Step 2/2: Visualizing occupancy"
echo "============================================================"
python "$SCRIPT_DIR/visualizers/vis_occ.py" "$@"

echo "============================================================"
echo "  All visualizations complete!"
echo "============================================================"
