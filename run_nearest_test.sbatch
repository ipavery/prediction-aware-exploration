#!/bin/bash -l
#SBATCH --job-name=PIPE-nearest
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=PIPE-nearest-%j.out
#SBATCH --error=PIPE-nearest-%j.err

set -eo pipefail

# Submit this job from the prediction-aware-exploration repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAP_ID="${MAP_ID:-}"
if [[ -z "$MAP_ID" ]]; then
    echo "ERROR: Set MAP_ID to one exact folder name under kth_test_maps."
    echo "Example: sbatch --export=ALL,MAP_ID=Train_building_floor_0 run_PIPE_nearest.sbatch"
    exit 1
fi

cd "$REPO_ROOT"
if [[ ! -f scripts/explore.py || ! -f configs/base.yaml ]]; then
    echo "ERROR: Submit from the prediction-aware-exploration repository root."
    exit 1
fi

MAP_DIR="$REPO_ROOT/kth_test_maps/$MAP_ID"
if [[ ! -f "$MAP_DIR/occ_map.npy" || ! -f "$MAP_DIR/valid_space.npy" ]]; then
    echo "ERROR: Expected this map structure: $MAP_DIR/{occ_map.npy,valid_space.npy}"
    exit 1
fi

# Initialize and activate the existing Conda environment.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pipe

# Rebuild range_libc for compatibility across cluster CPUs.
(
    set -e
    cd "$SLURM_SUBMIT_DIR/range_libc/pywrapper"

    # Keep a copy of the original build settings.
    cp -n setup.py setup.py.before-cpu-fix

    python -c 'from pathlib import Path; p = Path("setup.py"); text = p.read_text(); p.write_text(text.replace("-march=native", "-march=x86-64"))'

    # Force recompilation rather than reusing existing object files.
    WITH_CUDA=OFF python setup.py build_ext --force install
)

# Exercise the constructor and raycasting operation that crashed.
python -u -X faulthandler -c '
import numpy as np
import range_libc

print("Loaded:", range_libc.__file__, flush=True)

grid = np.ones((64, 64), dtype=np.uint8)
grid[1:-1, 1:-1] = 0

omap = range_libc.PyOMap(grid)
raycaster = range_libc.PyBresenhamsLine(omap, 20.0)

queries = np.array([[32.0, 32.0, 0.0]], dtype=np.float32)
ranges = np.zeros(1, dtype=np.float32)
raycaster.calc_range_many(queries, ranges)

assert np.isfinite(ranges).all()
print("Raycaster test passed:", ranges, flush=True)
'

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$TORCH_HOME"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Map: $MAP_ID"
echo "Repository: $REPO_ROOT"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# explore.py loads these checkpoints at startup even for nearest mode.
if [[ ! -f "$REPO_ROOT/pretrained_models/weights/big_lama/models/best.ckpt" ]]; then
    echo "ERROR: Missing pretrained_models/weights/big_lama/models/best.ckpt"
    echo "The current explore.py loads the big LaMa checkpoint even for nearest mode."
    exit 1
fi
if [[ ! -d "$REPO_ROOT/pretrained_models/weights/lama_ensemble" ]]; then
    echo "ERROR: Missing pretrained_models/weights/lama_ensemble"
    echo "The current base.yaml enables the LaMa ensemble, which explore.py loads at startup."
    exit 1
fi
shopt -s nullglob
ENSEMBLE_CHECKPOINTS=("$REPO_ROOT"/pretrained_models/weights/lama_ensemble/train_*/models/best.ckpt)
if [[ ${#ENSEMBLE_CHECKPOINTS[@]} -eq 0 ]]; then
    echo "ERROR: No train_*/models/best.ckpt files found under pretrained_models/weights/lama_ensemble"
    exit 1
fi

# Check the config before submitting compute time: this is a short nearest-only
# data-collection run, not an evaluation of every configured planner.
python - <<'PY'
from pathlib import Path
import yaml

with open("configs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

if cfg.get("modes_to_test") != ["nearest"]:
    raise SystemExit("ERROR: Set modes_to_test: ['nearest'] in configs/base.yaml")
if cfg.get("start_pose") is not None:
    raise SystemExit("ERROR: Set start_pose: null in configs/base.yaml for a random valid start")
if int(cfg.get("mission_time", 0)) > 200:
    raise SystemExit("ERROR: For this smoke run, set mission_time to 200 or less")
if Path(cfg.get("root_path", "")).expanduser().resolve() != Path.cwd().resolve():
    raise SystemExit("ERROR: configs/base.yaml root_path must point to this repository root")
print("Config OK: nearest only, random valid start, mission_time <= 200")
PY

# Verify that PyTorch can use the allocated GPU.
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see a CUDA GPU")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((128, 128), device="cuda")
print("GPU calculation result:", (x @ x).sum().item())
PY

# Run from scripts/ because explore.py locates configs relative to this cwd.
# Prevent its hard-coded TkAgg request from requiring a display on the node.
cd "$REPO_ROOT/scripts"
python -u - "$MAP_ID" <<'PY'
import runpy
import sys
import matplotlib

matplotlib.use("Agg")
# explore.py explicitly requests TkAgg; keep the already selected Agg backend.
matplotlib.use = lambda *args, **kwargs: None
map_id = sys.argv[1]
sys.argv = ["explore.py", "--collect_world_list", map_id]
runpy.run_path("explore.py", run_name="__main__")
PY

echo "Nearest-frontier run finished. Check $REPO_ROOT/experiments for gt_map.png, global_obs/, and run_viz/."
