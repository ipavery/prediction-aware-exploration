#!/bin/bash
# Source this file, or let submit_png_training.sh source it for you.
# Defaults below match the paths in job 15356429. Edit them once as needed.
# Existing exported values take precedence; explicitly empty BASE_CKPT means scratch.

export REPO="${REPO:-/home/ipa28/prediction-aware-exploration}"
export PNG_DATA_ROOT="${PNG_DATA_ROOT:-$REPO/1124_datacollect_v2_jobs_global_split_to_3}"
export PREPARED="${PREPARED:-$REPO/png_lama_prepared}"
export OUT="${OUT:-/home/ipa28/nobackup/autodelete/lama_png_scratch}"

export RESUME="${RESUME:-}"
if [[ -n "$RESUME" ]]; then
  export BASE_CKPT="${BASE_CKPT-}"
else
  export BASE_CKPT="${BASE_CKPT-$REPO/pretrained_models/places2_with_discriminator_weights/big-lama-with-discr-remove-loss_segm_pl.ckpt}"
fi
export BASE_DISCRIMINATOR="${BASE_DISCRIMINATOR:-true}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WORKERS="${WORKERS:-4}"
export SEED="${SEED:-42}"
export PAD_MAX="${PAD_MAX:-64}"
# Settings for the lama-h200 environment.
export PNG_GPU_TYPE="${PNG_GPU_TYPE:-h200}"
export PNG_TRAIN_QOS="${PNG_TRAIN_QOS:-eng}"
export PNG_SMOKE_QOS="${PNG_SMOKE_QOS:-eng}"
export PNG_CONDA_ENV="${PNG_CONDA_ENV:-lama-h200}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-lama-png}"
export PNG_USE_HTTP_PROXY="${PNG_USE_HTTP_PROXY:-1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
# prompt: ask for a key only in the interactive submitter, without saving it.
# login: use `wandb login` (reuses an existing login or prompts and saves SDK credentials).
export PNG_WANDB_AUTH="${PNG_WANDB_AUTH:-prompt}"
# Optional: export WANDB_ENTITY=your-wandb-user-or-team
# Optional Conda activation can go here, using your own Conda installation path.
# Never put the API key in this file. It contains no prompts and is safe to source in a job.
