#!/bin/bash

# test_blending_visual.sh - Test blending with Open3D visualization
# Tests PCDP calibration for D405

# Set library path for pinocchio compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set Python path to include project root
export PYTHONPATH=/home/leejungwook/point_cloud_blending/bpc_rise_clean:$PYTHONPATH

echo "======================================================================"
echo "Blending Visualization Test"
echo "======================================================================"
echo "This script will:"
echo "  1. Initialize Femto Bolt + D405 cameras"
echo "  2. Blend point clouds using PCDP calibration"
echo "  3. Visualize blended point cloud in Open3D"
echo ""
echo "Controls:"
echo "  SpaceMouse - Move robot to test different poses"
echo "  Q - Quit program"
echo "======================================================================"
echo ""

# Run 3-window visualization (Femto, D405, Blending)
# Pass any additional arguments (e.g., --verbose, --no-femto, etc.)
python pcdp/real_world/pc_visualize.py "$@"

echo ""
echo "======================================================================"
echo "Test completed."
echo "======================================================================"
