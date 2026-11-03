#!/bin/bash -l
#SBATCH --job-name=generate-lama-pairs
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=eng
#SBATCH --output=logs/lama-pairs-%j.out
#SBATCH --error=logs/lama-pairs-%j.err
#SBATCH --mail-user=ipa28@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# Device selection comes from configs/base.yaml (lama_device).
# For lama_device: 'cuda:0', also request a GPU at submission time:
#   sbatch --export=ALL,MAPS_DIR=/path/to/maps,DATA_ROOT=$HOME/nobackup/autodelete/lama_map_pairs collect_lama_pairs.sbatch
#   sbatch --gres=gpu:1 --export=ALL,MAPS_DIR=/path/to/maps,DATA_ROOT=$HOME/nobackup/autodelete/lama_map_pairs collect_lama_pairs.sbatch
# Specify the output folder name (SPLIT):
#   sbatch --export=ALL,MAPS_DIR=/home/ipa28/prediction-aware-exploration/maps_test_and_train/test_maps,DATA_ROOT=/home/ipa28/nobackup/autodelete/lama_map_pairs,SPLIT=test collect_lama_pairs.sbatch

set -euo pipefail
trap 'status=$?; echo "ERROR: collect_lama_pairs stopped at line ${LINENO} (exit ${status})" >&2' ERR

# Submit this job from the prediction-aware-exploration repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAPS_DIR="${MAPS_DIR:-}"
MAX_STEPS="${MAX_STEPS:-200}"
TARGET_KNOWN_FRACTION="${TARGET_KNOWN_FRACTION:-0.25}"
DATA_ROOT="${DATA_ROOT:-$HOME/nobackup/autodelete/lama_pairs_${SLURM_JOB_ID}}"
SPLIT="${SPLIT:-train_3}"

if [[ -z "$MAPS_DIR" || ! -d "$MAPS_DIR" ]]; then
    echo "ERROR: Set MAPS_DIR to a directory containing map subfolders." >&2
    echo "Example: sbatch --export=ALL,MAPS_DIR=/path/to/maps collect_lama_pairs.sbatch" >&2
    exit 1
fi
if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_STEPS must be a positive integer." >&2
    exit 1
fi
if [[ ! "$SPLIT" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: SPLIT must be a simple directory name." >&2
    exit 1
fi
MAPS_DIR="$(realpath "$MAPS_DIR")"

cd "$REPO_ROOT"
if [[ ! -f scripts/explore.py || ! -f configs/base.yaml ]]; then
    echo "ERROR: Submit from the prediction-aware-exploration repository root." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Conda activation hooks may inspect optional environment variables that are
# intentionally unset (for example MKL_INTERFACE_LAYER). Disable nounset only
# while those hooks run, then restore strict checking for the job script.
set +u
conda activate pipe
set -u

# This path planner uses the CPU. Build range_libc without its optional CUDA extension.
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

# Read mode and device selection from base.yaml. This remains the single source
# of truth; no custom explore.py command-line arguments are needed.
RUNTIME_CONFIG="$(python - <<'PY'
import yaml
import sys

with open("configs/base.yaml", "r") as stream:
    config = yaml.safe_load(stream)

if config.get("modes_to_test") != ["nearest"]:
    raise SystemExit("ERROR: Set modes_to_test: ['nearest'] in configs/base.yaml")
if config.get("start_pose") is not None:
    raise SystemExit("ERROR: Set start_pose: null in configs/base.yaml for random starts")

lama_device = config.get("lama_device", "cpu")
if lama_device not in ("cpu", "cuda:0"):
    raise SystemExit("ERROR: configs/base.yaml lama_device must be 'cpu' or 'cuda:0'")

model_modes = {"onlyvar", "visunk", "visvar", "mapex", "upen", "hectoraug", "pipe"}
needs_lama = bool(config.get("log_iou", False)) or any(
    mode in model_modes for mode in config["modes_to_test"])
print("{}\t{}".format(lama_device, "1" if needs_lama else "0"))
print("Config OK: nearest-only exploration, random start poses; lama_device={}".format(lama_device), file=sys.stderr)
PY
)"
if [[ "$RUNTIME_CONFIG" != *$'\t'* ]]; then
    echo "ERROR: Could not parse device/mode settings from configs/base.yaml." >&2
    exit 1
fi
LAMA_DEVICE="${RUNTIME_CONFIG%%$'\t'*}"
NEEDS_LAMA="${RUNTIME_CONFIG#*$'\t'}"
if [[ "$NEEDS_LAMA" != "0" && "$NEEDS_LAMA" != "1" ]]; then
    echo "ERROR: Unexpected mode flag from configs/base.yaml: $NEEDS_LAMA" >&2
    exit 1
fi
echo "Runtime settings parsed: lama_device=$LAMA_DEVICE, needs_lama=$NEEDS_LAMA"

if [[ "$NEEDS_LAMA" == "1" ]]; then
    if [[ "$LAMA_DEVICE" == "cuda:0" ]]; then
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            echo "ERROR: base.yaml selects cuda:0 but Slurm did not allocate a GPU; submit with --gres=gpu:1 or set lama_device: 'cpu'." >&2
            exit 1
        fi
        echo "Checking the CUDA device selected by configs/base.yaml..."
        python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot use the allocated GPU")
print("GPU:", torch.cuda.get_device_name(0))
try:
    x = torch.ones((16, 16), device="cuda")
    print("GPU check:", (x @ x).sum().item())
except Exception as error:
    raise SystemExit("ERROR: PyTorch cannot run on this GPU: {}".format(error))
PY
    else
        echo "LaMa inference will use CPU as selected in configs/base.yaml."
    fi
else
    # Nearest-only collection does not use LaMa. Do not initialize CUDA or run
    # the GPU smoke check, regardless of the configured lama_device.
    echo "Selected mode does not use LaMa; skipping model and GPU initialization."
fi

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
    echo "ERROR: No immediate map subdirectories found under $MAPS_DIR" >&2
    exit 1
fi

if [[ "$NEEDS_LAMA" == "1" ]]; then
    if [[ ! -f "$REPO_ROOT/pretrained_models/weights/big_lama/models/best.ckpt" ]]; then
        echo "ERROR: Missing pretrained_models/weights/big_lama/models/best.ckpt" >&2
        exit 1
    fi
    if [[ ! -d "$REPO_ROOT/pretrained_models/weights/lama_ensemble" ]]; then
        echo "ERROR: Missing pretrained_models/weights/lama_ensemble" >&2
        exit 1
    fi
    shopt -s nullglob
    ENSEMBLE_CHECKPOINTS=("$REPO_ROOT"/pretrained_models/weights/lama_ensemble/train_*/models/best.ckpt)
    if [[ ${#ENSEMBLE_CHECKPOINTS[@]} -eq 0 ]]; then
        echo "ERROR: No train_*/models/best.ckpt files found under pretrained_models/weights/lama_ensemble" >&2
        exit 1
    fi
fi

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Map folder: $MAPS_DIR"
echo "Maps / expected pairs: $MAP_COUNT"
echo "Step cap: $MAX_STEPS"
echo "Target known fraction: $TARGET_KNOWN_FRACTION"
echo "Split: $SPLIT"
echo "Training pairs: $DATA_ROOT/$SPLIT"
echo "LaMa required by selected mode: $NEEDS_LAMA"
echo "LaMa device from base.yaml: $LAMA_DEVICE"
echo "Experiment artifacts: disabled"
echo "Conda environment: $CONDA_DEFAULT_ENV"
if [[ "$NEEDS_LAMA" == "1" && "$LAMA_DEVICE" == "cuda:0" ]]; then
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
fi

# Run from scripts/ so explore.py resolves ../configs; Agg avoids Tk on compute nodes.
cd "$REPO_ROOT/scripts"
python -u - "$MAPS_DIR" "$DATA_ROOT" "$SPLIT" "$MAX_STEPS" "$TARGET_KNOWN_FRACTION" <<'PY'
import matplotlib
import os
import runpy
import sys

import cv2

matplotlib.use("Agg")
matplotlib.use = lambda *args, **kwargs: None
maps_dir, data_root, split_name, max_steps, target_fraction = sys.argv[1:]
sys.argv = [
    "explore.py",
    "--map_dir", maps_dir,
    "--max_steps", max_steps,
    "--target_known_fraction", target_fraction,
    "--training_data_root", data_root,
    "--training_data_split", split_name,
    "--no_experiment_artifacts",
]
collection = runpy.run_path("explore.py", run_name="__main__")

expected_samples = [item["sample_name"] for item in collection["run_exploration_args"]]
failures = []
for sample_name in expected_samples:
    gt_path = os.path.join(data_root, split_name, "global_gt", sample_name + ".png")
    obs_path = os.path.join(data_root, split_name, "global_obs", sample_name + ".png")
    gt = cv2.imread(gt_path) if os.path.isfile(gt_path) else None
    obs = cv2.imread(obs_path) if os.path.isfile(obs_path) else None
    if gt is None or obs is None:
        failures.append(sample_name + ": missing or unreadable GT/observation")
    elif gt.shape != obs.shape:
        failures.append(sample_name + ": GT/observation dimensions differ")
if failures:
    raise SystemExit("ERROR: Incomplete collection:\n" + "\n".join(failures))
print("Validated {} pairs from collection {}".format(
    len(expected_samples), collection["collection_id"]))
PY

echo "LaMa pair collection finished. Data: $DATA_ROOT/$SPLIT"
echo "Expected pairs: $MAP_COUNT"

