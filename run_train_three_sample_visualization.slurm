#!/bin/bash
#SBATCH --job-name=kite_train_vis
#SBATCH --partition=p3
#SBATCH --output=/home_cu/e01547/gemma-4-31B-it/slurm/kite/kite_train_vis_%j.out
#SBATCH --error=/home_cu/e01547/gemma-4-31B-it/slurm/kite/kite_train_vis_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

USER_NAME="e01547"
GEMMA_DIR="/home_cu/${USER_NAME}/gemma-4-31B-it"
ALPAMAYO_DIR="/home_cu/${USER_NAME}/alpamayo"
KITE_DIR="${GEMMA_DIR}/slurm/kite"
SIF_PATH="/home_cu/${USER_NAME}/gemma-cu13-kitscenes-v2.sif"
DOCKER_URI="docker://iwloci-registry01.intern.th-ab.de:5000/kav_e01547_gemma:cu130"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START="${KITSCENES_TRAIN_START:-0}"
LIMIT="${KITSCENES_TRAIN_LIMIT:-3}"
PARQUET="${KITSCENES_TRAIN_PARQUET:-${ALPAMAYO_DIR}/kitscenes-data/data/train-00000-of-00002.parquet}"
OUTPUT_ROOT="${KITE_TRAIN_OUTPUT_ROOT:-${GEMMA_DIR}/outputs/kite_train_three_samples}"
PLOT_DIR="${OUTPUT_ROOT}/three_method_train_mms_plots"

echo "Running on $(hostname)"
nvidia-smi

if [ ! -f "${SIF_PATH}" ]; then
  echo "SIF not found, pulling Gemma Docker image..."
  cd "/home_cu/${USER_NAME}"
  apptainer pull --name "$(basename "${SIF_PATH}")" "${DOCKER_URI}"
else
  echo "using existing ${SIF_PATH}"
fi

echo "Train parquet: ${PARQUET}"
echo "start=${START}"
echo "limit=${LIMIT}"
echo "output_root=${OUTPUT_ROOT}"
echo "plot_dir=${PLOT_DIR}"

apptainer exec --nv \
  --bind "${GEMMA_DIR}:${GEMMA_DIR}" \
  --bind "${ALPAMAYO_DIR}:${ALPAMAYO_DIR}" \
  "${SIF_PATH}" \
  python3 "${KITE_DIR}/run_all_kite.py" \
    --config "${KITE_DIR}/config.json" \
    --model-root "${GEMMA_DIR}" \
    --parquet "${PARQUET}" \
    --output-root "${OUTPUT_ROOT}" \
    --start "${START}" \
    --limit "${LIMIT}"

apptainer exec \
  --bind "${GEMMA_DIR}:${GEMMA_DIR}" \
  --bind "${ALPAMAYO_DIR}:${ALPAMAYO_DIR}" \
  "${SIF_PATH}" \
  python3 "${KITE_DIR}/plot_three_method_camera_trajectories.py" \
    --parquet "${PARQUET}" \
    --direct-dir "${OUTPUT_ROOT}/kitscenes_test_kite_direct_waypoints_egohistory" \
    --language-dir "${OUTPUT_ROOT}/kitscenes_test_kite_language_actions_bicycle" \
    --geometry-jsonl "${OUTPUT_ROOT}/kitscenes_test_kite_geometry_actions_bicycle_submission_with_reasoning.jsonl" \
    --geometry-dir "${OUTPUT_ROOT}/kitscenes_test_kite_geometry_actions_bicycle" \
    --output-dir "${PLOT_DIR}" \
    --start "${START}" \
    --limit "${LIMIT}"

echo "DONE"
echo "Plots:"
ls -lh "${PLOT_DIR}"/*.png "${PLOT_DIR}/plot_summary.json"
