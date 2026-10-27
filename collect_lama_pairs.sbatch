#!/bin/bash -l
#SBATCH --job-name=lama-pairs
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=logs/lama-pairs-%j.out
#SBATCH --error=logs/lama-pairs-%j.err
#SBATCH --mail-user=ipa28@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# usage: sbatch --export=ALL,MAPS_DIR=/home/ipa28/prediction-aware-exploration/train_maps,DATA_ROOT="$HOME/nobackup/autodelete/lama_map_pairs" collect_lama_pairs.sbatch

set -eo pipefail

# Submit this job from the prediction-aware-exploration repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAPS_DIR="${MAPS_DIR:-}"
MAX_STEPS="${MAX_STEPS:-200}"
TARGET_KNOWN_FRACTION="${TARGET_KNOWN_FRACTION:-0.25}"
DATA_ROOT="${DATA_ROOT:-$HOME/nobackup/autodelete/lama_pairs_${SLURM_JOB_ID}}"

if [[ -z "$MAPS_DIR" || ! -d "$MAPS_DIR" ]]; then
    echo "ERROR: Set MAPS_DIR to a directory containing map subfolders."
    echo "Example: sbatch --export=ALL,MAPS_DIR=/path/to/map_folder collect_lama_pairs.sbatch"
    exit 1
fi
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_STEPS must be a positive integer."
    exit 1
fi
MAPS_DIR="$(realpath "$MAPS_DIR")"

cd "$REPO_ROOT"
if [[ ! -f scripts/explore.py || ! -f configs/base.yaml ]]; then
    echo "ERROR: Submit from the prediction-aware-exploration repository root."
    exit 1
fi

# Initialize and activate the existing Conda environment.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pipe

# Rebuild range_libc for compatibility across cluster CPUs.
(
    set -e
    cd "$REPO_ROOT/range_libc/pywrapper"
    cp -n setup.py setup.py.before-cpu-fix
    python -c 'from pathlib import Path; p = Path("setup.py"); text = p.read_text(); p.write_text(text.replace("-march=native", "-march=x86-64"))'
    WITH_CUDA=OFF python setup.py build_ext --force install
)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$TORCH_HOME" "$DATA_ROOT"

# explore.py loads LaMa checkpoints at startup even for nearest-frontier mode.
if [[ ! -f "$REPO_ROOT/pretrained_models/weights/big_lama/models/best.ckpt" ]]; then
    echo "ERROR: Missing pretrained_models/weights/big_lama/models/best.ckpt"
    exit 1
fi
if [[ ! -d "$REPO_ROOT/pretrained_models/weights/lama_ensemble" ]]; then
    echo "ERROR: Missing pretrained_models/weights/lama_ensemble"
    exit 1
fi
shopt -s nullglob
ENSEMBLE_CHECKPOINTS=("$REPO_ROOT"/pretrained_models/weights/lama_ensemble/train_*/models/best.ckpt)
if [[ ${#ENSEMBLE_CHECKPOINTS[@]} -eq 0 ]]; then
    echo "ERROR: No train_*/models/best.ckpt files found under pretrained_models/weights/lama_ensemble"
    exit 1
fi

# Keep this focused on nearest-frontier data collection with random start poses.
python - <<'PY'
import yaml

with open("configs/base.yaml", "r") as stream:
    config = yaml.safe_load(stream)

if config.get("modes_to_test") != ["nearest"]:
    raise SystemExit("ERROR: Set modes_to_test: ['nearest'] in configs/base.yaml")
if config.get("start_pose") is not None:
    raise SystemExit("ERROR: Set start_pose: null in configs/base.yaml for random starts")
print("Config OK: nearest-only exploration, random start poses")
PY

MAP_COUNT="$(python - "$MAPS_DIR" <<'PY'
import os
import sys

root = sys.argv[1]
maps = [
    entry.name for entry in os.scandir(root)
    if entry.is_dir()
    and os.path.isfile(os.path.join(entry.path, "occ_map.npy"))
    and os.path.isfile(os.path.join(entry.path, "valid_space.npy"))
]
print(len(maps))
PY
)"
if [[ "$MAP_COUNT" -lt 1 ]]; then
    echo "ERROR: No immediate map subdirectories found under $MAPS_DIR"
    exit 1
fi
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see a CUDA GPU")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((128, 128), device="cuda")
print("GPU calculation result:", (x @ x).sum().item())
PY

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Map folder: $MAPS_DIR"
echo "Maps / expected pairs: $MAP_COUNT"
echo "Step cap: $MAX_STEPS"
echo "Target known fraction: $TARGET_KNOWN_FRACTION"
echo "Experiment artifacts: disabled"
echo "Training pairs: $DATA_ROOT/train_3"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Run from scripts/ so explore.py resolves ../configs; Agg avoids Tk on compute nodes.
cd "$REPO_ROOT/scripts"
python -u - "$MAPS_DIR" "$DATA_ROOT" "$MAX_STEPS" "$TARGET_KNOWN_FRACTION" <<'PY'
import matplotlib
import os
import runpy
import sys

import cv2

matplotlib.use("Agg")
matplotlib.use = lambda *args, **kwargs: None
maps_dir, data_root, max_steps, target_fraction = sys.argv[1:]
sys.argv = [
    "explore.py",
    "--map_dir", maps_dir,
    "--max_steps", max_steps,
    "--target_known_fraction", target_fraction,
    "--training_data_root", data_root,
    "--no_experiment_artifacts",
]
collection = runpy.run_path("explore.py", run_name="__main__")

# Validate exactly this invocation's samples, even when DATA_ROOT has older pairs.
expected_samples = [item['sample_name'] for item in collection['run_exploration_args']]
failures = []
for sample_name in expected_samples:
    gt_path = os.path.join(data_root, 'train_3', 'global_gt', sample_name + '.png')
    obs_path = os.path.join(data_root, 'train_3', 'global_obs', sample_name + '.png')
    gt = cv2.imread(gt_path) if os.path.isfile(gt_path) else None
    obs = cv2.imread(obs_path) if os.path.isfile(obs_path) else None
    if gt is None or obs is None:
        failures.append(sample_name + ': missing or unreadable GT/observation')
    elif gt.shape != obs.shape:
        failures.append(sample_name + ': GT/observation dimensions differ')
if failures:
    raise SystemExit('ERROR: Incomplete collection:\n' + '\n'.join(failures))
print('Validated {} pairs from collection {}'.format(
    len(expected_samples), collection['collection_id']))
PY

echo "LaMa pair collection finished. Data: $DATA_ROOT/train_3"
echo "Expected pairs: $MAP_COUNT"
