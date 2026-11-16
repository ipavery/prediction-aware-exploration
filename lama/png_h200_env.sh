#!/bin/bash
# Keep the user's existing dataset/checkpoint/output/W&B settings.
png_h200_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "$png_h200_dir/png_training_env.sh" ]]; then
  cp -n -- "$png_h200_dir/png_training_env.sh.example" "$png_h200_dir/png_training_env.sh"
  printf 'Created %s/png_training_env.sh from the included defaults.\n' "$png_h200_dir" >&2
fi
source "$png_h200_dir/png_training_env.sh"
export PNG_CONDA_ENV=lama-h200
export PNG_GPU_TYPE=h200
export PNG_TRAIN_QOS=eng
export PNG_SMOKE_QOS=eng
unset png_h200_dir
