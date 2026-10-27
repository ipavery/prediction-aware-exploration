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
parser.add_argument(
    "--mode",
    choices=["full", "occ-only"],
    default="full",
    help="Render full occupancy+valid-space preview or just the raw occupancy map.",
)
parser.add_argument(
    "--count",
    type=int,
    default=10,
    help="Number of maps to sample for rendering.",
)
args = parser.parse_args()

map_folders = [
    folder
    for folder in args.folder.iterdir()
    if folder.is_dir()
    and (folder / "occ_map.npy").is_file()
    and (folder / "valid_space.npy").is_file()
]
if len(map_folders) < args.count:
    raise ValueError(f"Expected at least {args.count} maps in {args.folder}, found {len(map_folders)}")

output_folder = Path(__file__).resolve().parent / "map_previews"
output_folder.mkdir(exist_ok=True)

selected_folders = random.sample(map_folders, args.count)

for folder in selected_folders:
    occ = np.load(folder / "occ_map.npy")
    valid_space = np.load(folder / "valid_space.npy")

    if args.mode == "occ-only":
        fig, ax = plt.subplots(figsize=(occ.shape[1] / 100.0, occ.shape[0] / 100.0), dpi=100)
        ax.imshow(occ, cmap="gray", interpolation="nearest")
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        plt.subplots_adjust(0, 0, 1, 1)
        output = output_folder / f"{folder.name}_occ_only.png"
        fig.savefig(output, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved occupancy-only preview to: {output} (shape={occ.shape})")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
    axes[0].imshow(occ, cmap="gray", interpolation="nearest")
    axes[0].set_title("Occupancy")
    axes[1].imshow(valid_space > 0, cmap="gray", interpolation="nearest")
    axes[1].set_title("Valid space")
    for ax in axes:
        ax.axis("off")

    output = output_folder / f"{folder.name}.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved preview to: {output} (shape={occ.shape})")