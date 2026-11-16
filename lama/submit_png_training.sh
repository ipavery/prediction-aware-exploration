#!/bin/bash
set -euo pipefail
png_submit_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
png_mode=${1:-smoke}
if [[ $# -gt 0 ]]; then shift; fi
case "$png_mode" in
  check|smoke|train) ;;
  *) echo 'Usage: bash submit_png_training.sh [check|smoke|train] [additional sbatch options]' >&2; exit 2 ;;
esac
source "${PNG_TRAINING_ENV:-$png_submit_dir/png_training_env.sh}"
source "$png_submit_dir/png_activate_conda.sh"

for png_required_dir in "$REPO/lama" "$PNG_DATA_ROOT/train_1/global_gt" "$PNG_DATA_ROOT/train_1/global_obs"; do
  if [[ ! -d "$png_required_dir" ]]; then
    printf 'Missing directory: %s\nEdit png_training_env.sh or export the correct path.\n' "$png_required_dir" >&2
    exit 2
  fi
done
for png_required_file in "$REPO/lama/train_png_compare.sbatch" "$PREPARED/train.json" "$PREPARED/validate.json"; do
  if [[ ! -f "$png_required_file" ]]; then printf 'Missing file: %s\n' "$png_required_file" >&2; exit 2; fi
done
if [[ -n "$BASE_CKPT" && -n "$RESUME" ]]; then
  echo 'Set BASE_CKPT for initialization or RESUME for continuation, not both.' >&2; exit 2
fi
for png_checkpoint in "$BASE_CKPT" "$RESUME"; do
  if [[ -n "$png_checkpoint" && ! -f "$png_checkpoint" ]]; then
    printf 'Missing checkpoint: %s\n' "$png_checkpoint" >&2; exit 2
  fi
done
if [[ "$png_mode" == check ]]; then
  printf 'Paths checked. GPU request: %s; W&B mode: %s; proxy enabled for online jobs: %s\n' "${PNG_GPU_TYPE:-p100}" "$WANDB_MODE" "$PNG_USE_HTTP_PROXY"
  exit 0
fi
if [[ "$WANDB_MODE" == online && -z ${WANDB_API_KEY:-} ]]; then
  case "$PNG_WANDB_AUTH" in
    prompt)
      if [[ ! -t 0 ]]; then
        echo 'No W&B API key and no interactive terminal. Export the key, use PNG_WANDB_AUTH=login, or WANDB_MODE=offline.' >&2
        exit 2
      fi
      read -rp 'W&B API key: ' WANDB_API_KEY
      printf '\n'
      if [[ -z "$WANDB_API_KEY" ]]; then echo 'No API key entered.' >&2; exit 2; fi
      export WANDB_API_KEY
      ;;
    login) wandb login ;;
    *) echo 'PNG_WANDB_AUTH must be prompt or login.' >&2; exit 2 ;;
  esac
fi
cd "$REPO/lama"
mkdir -p logs
png_resource_options=("--gres=gpu:${PNG_GPU_TYPE:-p100}:1")
if [[ "$png_mode" == smoke ]]; then
  export SMOKE=1
  sbatch --export=ALL "${png_resource_options[@]}" "--qos=${PNG_SMOKE_QOS:-${PNG_TRAIN_QOS:-test}}" --time=01:00:00 "$@" train_png_compare.sbatch
else
  export SMOKE=0
  if [[ -n ${PNG_TRAIN_QOS:-} ]]; then
    png_resource_options+=("--qos=$PNG_TRAIN_QOS")
  fi
  sbatch --export=ALL "${png_resource_options[@]}" "$@" train_png_compare.sbatch
fi
# No retries: check squeue/sacct first if submission times out.
# A prompted key existed only in this child process and the submitted job environment.
