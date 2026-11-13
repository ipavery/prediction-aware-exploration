#!/bin/bash -l
#SBATCH --job-name=lama-png-compare
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/png-compare-%j.out
#SBATCH --error=logs/png-compare-%j.err

set -euo pipefail
REPO=${REPO:-/home/ipa28/prediction-aware-exploration}
PREPARED=${PREPARED:-$REPO/png_lama_prepared}
OUT=${OUT:-/home/ipa28/nobackup/autodelete/lama_png_scratch}
: "${PNG_DATA_ROOT:?Set PNG_DATA_ROOT to the folder containing train_1 and test}"
: "${NO_SAT_CKPT:?Set NO_SAT_CKPT to the chosen no-satellite checkpoint}"
: "${SAT_CKPT:?Set SAT_CKPT to the chosen satellite checkpoint}"
SPLIT=${SPLIT:-validate}
if [[ "$SPLIT" != validate && "$SPLIT" != test ]]; then
  echo 'SPLIT must be validate or test' >&2; exit 2
fi
cd "$REPO/lama"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u bin/evaluate_png_maps.py \
  --manifest "$PREPARED/$SPLIT.json" --data-root "$PNG_DATA_ROOT" \
  --model nosatellite "$NO_SAT_CKPT" --model satellite "$SAT_CKPT" \
  --padding-max 0 64 --seed "${SEED:-42}" \
  --output "$OUT/comparisons/${SPLIT}_${SLURM_JOB_ID:-manual}"
