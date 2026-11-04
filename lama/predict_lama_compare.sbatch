#!/bin/bash -l
#SBATCH --job-name=lama-predict-compare
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=logs/lama-predict-compare-%j.out
#SBATCH --error=logs/lama-predict-compare-%j.err
#SBATCH --mail-user=ipa28@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# usage:
# cd /home/ipa28/prediction-aware-exploration/lama
# sbatch predict_lama_compare.sbatch

set -euo pipefail

# Update SATELLITE_MODEL to the Hydra run folder that contains config.yaml and
# models/best.ckpt (or change CHECKPOINT below if both model dirs use another name).
REPO=/home/ipa28/prediction-aware-exploration
DATA=/home/ipa28/nobackup/autodelete/lama_map_pairs
OUTPUT_ROOT=/home/ipa28/nobackup/autodelete/lama_satellite_training/predictions
BASE_MODEL="/home/ipa28/nobackup/autodelete/lama_regular_training/experiments/ipa28_2026-11-02_22-50-10_train_big-lama_"
SATELLITE_MODEL=/home/ipa28/nobackup/autodelete/lama_satellite_training/experiments/ipa28_2026-10-31_22-39-29_train_big-lama-satellite_satellite
CHECKPOINT=best.ckpt

cd "$REPO/lama"
export PYTHONPATH="$REPO/lama${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$HOME/.cache/torch"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate pipe
set -u

for model_dir in "$BASE_MODEL" "$SATELLITE_MODEL"; do
    if [[ ! -f "$model_dir/config.yaml" ]]; then
        echo "ERROR: Missing model config: $model_dir/config.yaml" >&2
        exit 1
    fi
    if [[ ! -f "$model_dir/models/$CHECKPOINT" ]]; then
        echo "ERROR: Missing checkpoint: $model_dir/models/$CHECKPOINT" >&2
        exit 1
    fi
done

for split_dir in global_gt global_obs global_satellite; do
    if [[ ! -d "$DATA/test/$split_dir" ]]; then
        echo "ERROR: Missing test data directory: $DATA/test/$split_dir" >&2
        exit 1
    fi
done

PREDICTIONS="$OUTPUT_ROOT/compare_${SLURM_JOB_ID}"
mkdir -p "$PREDICTIONS"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Base model: $BASE_MODEL"
echo "Satellite model: $SATELLITE_MODEL"
echo "Test data: $DATA/test"
echo "Predictions: $PREDICTIONS"

python -u bin/predict.py -cn map-satellite \
    "model.paths=['$BASE_MODEL','$SATELLITE_MODEL']" \
    "model.display_names=['Regular LaMa','Satellite LaMa']" \
    model.checkpoint="$CHECKPOINT" \
    indir="$DATA/test" \
    outdir="$PREDICTIONS" \
    dataset.use_satellite=true \
    dataset.num_frames_to_skip=1 \
    device=cuda \
    refine=false

echo "Prediction comparison complete: $PREDICTIONS"
