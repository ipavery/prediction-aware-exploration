#!/bin/bash -l
#SBATCH --job-name=lama-smoke
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=lama-smoke-%j.out
#SBATCH --error=lama-smoke-%j.err

set -eo pipefail

# Submit this job from the lama directory.
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f bin/train.py ]]; then
    echo "ERROR: Submit this job from the lama directory containing bin/train.py"
    exit 1
fi

# Initialize and activate the existing Conda environment.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pipe

export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$TORCH_HOME"

# Keep this run's data and output in BYU scratch, separate from the PSC paths.
JOB_ROOT="$HOME/nobackup/autodelete/lama_smoke_${SLURM_JOB_ID}"
DATA="$JOB_ROOT/data"
OUT="$JOB_ROOT/output"
mkdir -p "$DATA" "$OUT/experiments" "$OUT/tb_logs"

export JOB_DATA="$DATA"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Confirm that this job can run a computation on its allocated GPU.
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see a CUDA GPU")

x = torch.ones((512, 512), device="cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU calculation result:", (x @ x).sum().item())
PY

# Create paired synthetic ground-truth and observation images.
# The LaMa map loader treats pixels with value 128 in global_obs as unknown.
python - <<'PY'
import os
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(os.environ["JOB_DATA"])
rng = np.random.default_rng(7)

for split, count in (("train_3", 2), ("test", 2)):
    gt_dir = root / split / "global_gt"
    obs_dir = root / split / "global_obs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        gt = np.zeros((256, 256, 3), dtype=np.uint8)
        gt[32:224, 120:136] = 255
        gt[120:136, 32:224] = 255

        y, x = rng.integers(24, 168, size=2)
        obs = gt.copy()
        obs[y:y + 64, x:x + 64] = 128

        name = f"smoke_{i:04d}.png"
        Image.fromarray(gt).save(gt_dir / name)
        Image.fromarray(obs).save(obs_dir / name)

print("Synthetic train and test images created under", root)
PY

export HYDRA_FULL_ERROR=1 # So we can see all the errors

# Run a single small training batch on the one GPU allocated above.
python -u bin/train.py -cn big-lama \
    run_title=slurm_smoke \
    location.data_root_dir="$DATA" \
    location=kth_train_3 \
    location.out_root_dir="$OUT/experiments" \
    location.tb_dir="$OUT/tb_logs" \
    data.train.num_frames_to_skip=1 \
    data.val.num_frames_to_skip=1 \
    data.batch_size=2 \
    data.val_batch_size=1 \
    data.num_workers=0 \
    trainer.kwargs.gpus=1 \
    trainer.kwargs.max_epochs=1 \
    trainer.kwargs.limit_train_batches=1 \
    +trainer.kwargs.limit_val_batches=0 \
    trainer.kwargs.num_sanity_val_steps=0 \
    trainer.kwargs.val_check_interval=1 \
    trainer.checkpoint_kwargs.save_top_k=0 \
    trainer.checkpoint_kwargs.save_last=false \
    +evaluator.fid=false \
    +evaluator.lpips=false \
    evaluator.integral_kind=null

echo "LaMa smoke test finished."
echo "Outputs are under: $JOB_ROOT"