#!/bin/bash
#SBATCH --job-name=gemma_ablate_all
#SBATCH --partition=p2
#SBATCH --output=/home_cu/e01547/gemma-4-31B-it/slurm/kite/gemma_ablate_all_%j.out
#SBATCH --error=/home_cu/e01547/gemma-4-31B-it/slurm/kite/gemma_ablate_all_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

USER_NAME="e01547"
GEMMA_DIR="/home_cu/${USER_NAME}/gemma-4-31B-it"
ALPAMAYO_DIR="/home_cu/${USER_NAME}/alpamayo"
SIF_PATH="/home_cu/${USER_NAME}/gemma-cu13-kitscenes-v2.sif"
DOCKER_URI="docker://iwloci-registry01.intern.th-ab.de:5000/kav_e01547_gemma:cu130"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START="${KITSCENES_START_IDX:-0}"
LIMIT="${KITSCENES_LIMIT:-400}"
ABLATION_DIR="${GEMMA_DIR}/slurm/kite"
CONFIG="${ABLATION_DIR}/config.json"
OUTPUT_ROOT="${KITE_OUTPUT_ROOT:-${GEMMA_DIR}/outputs/kite}"
PARQUET="${ALPAMAYO_DIR}/kitscenes-data/data/test-00000-of-00001.parquet"

EXPERIMENTS=(
  direct_waypoints_egohistory
  direct_waypoints_motion
  language_actions_bicycle
  geometry_actions_bicycle
)

echo "Running on $(hostname)"
nvidia-smi

if [ ! -f "${SIF_PATH}" ]; then
  echo "SIF not found, pulling updated Gemma Docker image..."
  cd /home_cu/${USER_NAME}
  apptainer pull --name "$(basename "${SIF_PATH}")" "${DOCKER_URI}"
else
  echo "using existing ${SIF_PATH}"
fi

echo "Running all KITScenes ablations overnight"
echo "start=${START}"
echo "limit=${LIMIT}"
echo "output_root=${OUTPUT_ROOT}"
echo "parquet=${PARQUET}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  echo "============================================================"
  echo "Starting experiment=${EXPERIMENT} at $(date)"
  echo "============================================================"

  apptainer exec --nv \
    --bind "${GEMMA_DIR}:${GEMMA_DIR}" \
    --bind "${ALPAMAYO_DIR}:${ALPAMAYO_DIR}" \
    "${SIF_PATH}" \
    python3 "${ABLATION_DIR}/run_ablation.py" \
      --config "${CONFIG}" \
      --experiment "${EXPERIMENT}" \
      --model-root "${GEMMA_DIR}" \
      --parquet "${PARQUET}" \
      --output-root "${OUTPUT_ROOT}" \
      --start "${START}" \
      --limit "${LIMIT}"

  echo "Finished experiment=${EXPERIMENT} at $(date)"
done

echo "All ablations complete."
echo "Submission JSONLs:"
ls -lh "${OUTPUT_ROOT}"/*_submission.jsonl "${OUTPUT_ROOT}"/*_submission_with_reasoning.jsonl
