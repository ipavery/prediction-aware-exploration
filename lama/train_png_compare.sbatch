#!/bin/bash -l
#SBATCH --job-name=lama-png-train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=eng
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
cd "$REPO/lama"
source "$REPO/lama/png_activate_conda.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-lama-png}"
if [[ "$WANDB_MODE" == online && "${PNG_USE_HTTP_PROXY:-1}" == 1 ]]; then
  if ! type module >/dev/null 2>&1; then
    echo 'Online W&B needs BYU module load http_proxy. Use a login-shell job or WANDB_MODE=offline.' >&2
    exit 2
  fi
  module load http_proxy
  # Keep local W&B service communication off the external HTTP proxy.
  png_proxy_bypass=${no_proxy:-${NO_PROXY:-}}
  export no_proxy="localhost,127.0.0.1,::1${png_proxy_bypass:+,$png_proxy_bypass}"
  export NO_PROXY="$no_proxy"
  unset png_proxy_bypass
  export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
fi
export TORCH_HOME="$HOME/.cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) MODEL=nosatellite; SAT=false; CHANNELS=4 ;;
  1) MODEL=satellite; SAT=true; CHANNELS=5 ;;
  *) echo 'Array task must be 0 or 1' >&2; exit 2 ;;
esac
INIT=scratch
if [[ -n ${BASE_CKPT:-} ]]; then
  if [[ -n ${RESUME:-} ]]; then
    echo 'Use BASE_CKPT for a new fine-tune, or RESUME for continuation; not both.' >&2
    exit 2
  fi
  if [[ ! -f "$BASE_CKPT" ]]; then
    echo 'BASE_CKPT does not point to an existing checkpoint file.' >&2
    exit 2
  fi
  INIT=base
elif [[ -n ${RESUME:-} ]]; then
  INIT=resume
fi
TAG=${MODEL}_${INIT}_grayobs_blackprior_pad${PAD_MAX}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-png_${INIT}_pad${PAD_MAX}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUT/experiments" "$OUT/tb_logs"
OPTIONS=()
if [[ -n ${BASE_CKPT:-} ]]; then
  OPTIONS+=("pretrained_generator_path=$BASE_CKPT" "pretrained_discriminator=${BASE_DISCRIMINATOR:-auto}")
fi
if [[ "$SMOKE" == 1 ]]; then
  OPTIONS+=(trainer.kwargs.max_epochs=1 trainer.kwargs.limit_train_batches=2
            trainer.kwargs.limit_val_batches=2 trainer.kwargs.num_sanity_val_steps=0
            data.num_workers=0 png_data.val_limit=8)
fi
if [[ -n ${RESUME:-} ]]; then
  OPTIONS+=("trainer.kwargs.resume_from_checkpoint=$RESUME")
fi
python -u bin/check_png_training_gpu.py
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
  "data.batch_size=${BATCH_SIZE:-1}" "data.num_workers=${WORKERS:-4}" \
  "trainer.kwargs.max_epochs=${EPOCHS:-30}" \
  "${OPTIONS[@]}"
