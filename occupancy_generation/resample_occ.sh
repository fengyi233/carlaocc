#!/bin/bash
# Resample occupancy data to different voxel sizes.
# Usage: bash occupancy_generation/resample_occ.sh [hydra overrides...]
#   e.g. bash occupancy_generation/resample_occ.sh town_names='["Town05_Opt"]'

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  Step 1/2: Resampling occupancy: vs_0_2_forward_view"
echo "============================================================"
python "$SCRIPT_DIR/generators/resample_occ.py" \
  target.voxel_size=0.2 \
  target.save_dir_name=vs_0_2_forward_view \
  target.voxel_origin=[0,-25.6,-2.4] \
  target.volume_size=[256,256,32] \
  "$@"

echo "============================================================"
echo "  Step 2/2: Resampling occupancy: vs_0_4_surround_view"
echo "============================================================"
python "$SCRIPT_DIR/generators/resample_occ.py" \
  target.voxel_size=0.4 \
  target.save_dir_name=vs_0_4_surround_view \
  target.voxel_origin=[-25.6,-25.6,-2.4] \
  target.volume_size=[128,128,16] \
  "$@"

echo "============================================================"
echo "  All resampling complete!"
echo "============================================================"