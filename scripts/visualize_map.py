import matplotlib
matplotlib.use("Agg")
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "kth_test_maps",
    help="Dataset folder containing map subfolders",
)
args = parser.parse_args()

map_folders = [
    folder
    for folder in args.folder.iterdir()
    if folder.is_dir()
    and not folder.name.lower().startswith(("train_", "test_", "validate_"))
    and (folder / "occ_map.npy").is_file()
    and (folder / "valid_space.npy").is_file()
]
if len(map_folders) < 10:
    raise ValueError(f"Expected at least 10 maps in {args.folder}, found {len(map_folders)}")

output_folder = Path(__file__).resolve().parent / "map_previews"
output_folder.mkdir(exist_ok=True)

for folder in random.sample(map_folders, 10):
    occ = np.load(folder / "occ_map.npy")
    valid_space = np.load(folder / "valid_space.npy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(occ, cmap="gray")
    axes[0].set_title("Occupancy")
    axes[1].imshow(valid_space > 0, cmap="gray")
    axes[1].set_title("Valid space")
    for ax in axes:
        ax.axis("off")

    output = output_folder / f"{folder.name}.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview to: {output}")