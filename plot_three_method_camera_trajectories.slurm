#!/bin/bash
#SBATCH --job-name=plot_three_methods
#SBATCH --partition=p3
#SBATCH --output=/home_cu/e01547/gemma-4-31B-it/slurm/kite/plot_three_methods_%j.out
#SBATCH --error=/home_cu/e01547/gemma-4-31B-it/slurm/kite/plot_three_methods_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

USER_NAME="e01547"
GEMMA_DIR="/home_cu/${USER_NAME}/gemma-4-31B-it"
ALPAMAYO_DIR="/home_cu/${USER_NAME}/alpamayo"
SIF_PATH="/home_cu/${USER_NAME}/gemma-cu13-kitscenes-v2.sif"

START="${KITSCENES_START_IDX:-0}"
LIMIT="${KITSCENES_PLOT_LIMIT:-400}"
ABLATION_ROOT="${KITE_OUTPUT_ROOT:-${GEMMA_DIR}/outputs/kite}"
DIRECT_DIR="${ABLATION_ROOT}/kitscenes_test_ablation_direct_waypoints_egohistory"
LANGUAGE_DIR="${ABLATION_ROOT}/kitscenes_test_ablation_language_actions_bicycle"
GEOMETRY_JSONL="${GEMMA_DIR}/outputs/kitscenes_test_front3_last4_geometry_allsteer_savgol_reasoning_geomkin_submission_final_with_reasoning.jsonl"
GEOMETRY_DIR="${GEMMA_DIR}/outputs/kitscenes_test_front3_last4_geometry_allsteer_savgol_reasoning_geomkin"
OUTPUT_DIR="${ABLATION_ROOT}/three_method_camera_trajectory_plots"
PARQUET="${ALPAMAYO_DIR}/kitscenes-data/data/test-00000-of-00001.parquet"

echo "start=${START}"
echo "limit=${LIMIT}"
echo "output_dir=${OUTPUT_DIR}"

apptainer exec \
  --bind "${GEMMA_DIR}:${GEMMA_DIR}" \
  --bind "${ALPAMAYO_DIR}:${ALPAMAYO_DIR}" \
  "${SIF_PATH}" \
  python3 "${GEMMA_DIR}/slurm/kite/plot_three_method_camera_trajectories.py" \
    --parquet "${PARQUET}" \
    --direct-dir "${DIRECT_DIR}" \
    --language-dir "${LANGUAGE_DIR}" \
    --geometry-jsonl "${GEOMETRY_JSONL}" \
    --geometry-dir "${GEOMETRY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --start "${START}" \
    --limit "${LIMIT}"

echo "DONE"
