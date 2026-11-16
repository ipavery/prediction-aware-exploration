#!/bin/bash
set -euo pipefail
png_h200_submit_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PNG_TRAINING_ENV="$png_h200_submit_dir/png_h200_env.sh"
exec bash "$png_h200_submit_dir/submit_png_training.sh" "$@"
