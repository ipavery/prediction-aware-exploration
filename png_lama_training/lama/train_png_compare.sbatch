#!/bin/bash -l
#SBATCH --job-name=lama-png-scratch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=logs/png-lama-%A_%a.out
#SBATCH --error=logs/png-lama-%A_%a.err

set -euo pipefail
REPO=${REPO:-/home/ipa28/prediction-aware-exploration}
: "${PNG_DATA_ROOT:?Set PNG_DATA_ROOT to the extracted folder containing train_1 and test}"
PREPARED=${PREPARED:-$REPO/png_lama_prepared}
OUT=${OUT:-/home/ipa28/nobackup/autodelete/lama_png_scratch}
SEED=${SEED:-42}
PAD_MAX=${PAD_MAX:-64}
SMOKE=${SMOKE:-0}
# Activate the same Conda environment as your previous LaMa training before sbatch.
cd "$REPO/lama"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export TORCH_HOME="$HOME/.cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) MODEL=nosatellite; SAT=false; CHANNELS=4 ;;
  1) MODEL=satellite; SAT=true; CHANNELS=5 ;;
  *) echo 'Array task must be 0 or 1' >&2; exit 2 ;;
esac
TAG=${MODEL}_pad${PAD_MAX}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT/experiments" "$OUT/tb_logs"
OPTIONS=()
if [[ "$SMOKE" == 1 ]]; then
  OPTIONS+=(trainer.kwargs.max_epochs=1 trainer.kwargs.limit_train_batches=2
            trainer.kwargs.limit_val_batches=2 trainer.kwargs.num_sanity_val_steps=0
            data.num_workers=0 png_data.val_limit=8)
fi
if [[ -n ${RESUME:-} ]]; then
  OPTIONS+=("trainer.kwargs.resume_from_checkpoint=$RESUME")
fi
python -u bin/train_png_maps.py -cn png-maps \
  "run_title=$TAG" "seed=$SEED" \
  location=kth_train_3 \
  "location.data_root_dir=$PNG_DATA_ROOT" location.train_subdir=train_1 \
  "location.out_root_dir=$OUT/experiments" "location.tb_dir=$OUT/tb_logs" \
  "png_data.root=$PNG_DATA_ROOT" \
  "png_data.train_manifest=$PREPARED/train.json" \
  "png_data.val_manifest=$PREPARED/validate.json" \
  "png_data.padding_max=$PAD_MAX" \
  "training_model.condition_on_satellite=$SAT" "generator.input_nc=$CHANNELS" \
  "data.batch_size=${BATCH_SIZE:-2}" "data.num_workers=${WORKERS:-4}" \
  "trainer.kwargs.max_epochs=${EPOCHS:-30}" \
  "${OPTIONS[@]}"
