import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

folder = Path("/home/ipa28/prediction-aware-exploration/kth_test_maps/DatasetWithSplit2/kth_test_maps2/Test_asia_china_building_1203537175_floor_0")
occ = np.load(folder / "occ_map.npy")
valid_space = np.load(folder / "valid_space.npy")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(occ, cmap="gray")
axes[0].set_title("Occupancy")
axes[1].imshow(valid_space > 0, cmap="gray")
axes[1].set_title("Valid space")
for ax in axes:
    ax.axis("off")

output = Path(__file__).resolve().parent / "map_preview.png"
fig.savefig(output, dpi=150, bbox_inches="tight")
print(f"Saved preview to: {output}")