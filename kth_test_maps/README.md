# KTH test maps

KTH floor plans for the KTH part of the benchmark, one folder per map.

- `occ_map.npy`: 0 is wall, 254 is free.
- `valid_space.npy`: 0 is outside the building; any other value is a space ID.
- Both are stored at 0.05 m per pixel. The simulator halves them to its 0.1 m grid when it loads them (`scripts/sim_utils.py`).

These are the maps from the MapEx release, except that seven of them (`50010535_PLAN1`, `50010535_PLAN2`, `50010536_PLAN3`, `50015847`, `50015848`, `50037764_PLAN1`, `50052749`) have small wall gaps closed so the robot and lidar can't leave the building.

`50015847` and `50015848` were in the training data for the KTH LaMa models. We keep them for comparability with MapEx, but predictions on these two maps are better than on unseen maps, and IoU (computed on the prediction) is optimistic for every method.
