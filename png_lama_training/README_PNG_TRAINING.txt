LaMa scratch comparison on train_1 PNG pairs
==========================================

This bundle adds new files to ipavery/prediction-aware-exploration (isaac-test).
It does not change the existing exploration or training scripts. The original
PNG dataset stays where it is. No exploration run or NPY conversion is needed.

Your choices: use train_1 only; use simulated building-footprint priors.
The included png_lama_prepared manifests and 57 cached priors are ready to use.
Set PNG_DATA_ROOT on the cluster to the directory containing train_1 and test.
The manifests store relative image paths; that setting overrides the local
Windows source path recorded during preparation.

1. Install and run two short smoke jobs
-------------------------------------
Upload png_lama_training.zip to the cluster, then run in your EXISTING working
LaMa environment. Retain the isaac-test code with satellite support.

  cd /home/ipa28/prediction-aware-exploration
  unzip -n /path/to/png_lama_training.zip
  mkdir -p lama/logs
  export PNG_DATA_ROOT=/actual/extracted/dataset/folder
  test -d "$PNG_DATA_ROOT/train_1/global_gt"
  test -d "$PNG_DATA_ROOT/test/global_obs"
  cd lama
  SMOKE=1 sbatch --qos=test --time=01:00:00 train_png_compare.sbatch

The array starts one no-satellite job (task 0) and one satellite job (task 1).
Each uses one GPU. The smoke run uses two training batches, two validation
samples and one epoch. It should produce a checkpoint, metrics, and previews.
Check both jobs' logs in lama/logs and both exit statuses before a longer run.
The job scripts inherit your activated Conda environment. The production job
does not request a partition or QoS; add the settings required by your account.

2. Start a pilot, then continue to 30 epochs
------------------------------------------
After BOTH smoke jobs succeed:

  EPOCHS=5 sbatch train_png_compare.sbatch

Starting parameters:
  Architecture: existing big-LaMa, 18 FFC blocks, sigmoid RGB output.
  Inputs: 4 channels without satellite; 5 with the footprint prior.
  Initialization: generator and discriminator both random, no checkpoint load.
  Data: actual observed PNG as input; binary GT PNG as target.
  Native-scale crop: 512 x 512, no interpolation or resizing of map labels.
  Padding: four independently sampled integer margins from 0 through 64 pixels.
  Batch: 2 per GPU. Workers: 4. Precision: float32.
  Adam learning rates: generator 0.0001; discriminator 0.0001.
  L1: unknown weight 10, known weight 1, artificial padding excluded.
  LaMa adversarial weight 10; feature matching weight 100.
  VGG/ResNet perceptual losses disabled; no pretrained feature weights downloaded.
  Satellite-only training augmentation: shift up to 2 pixels, 10% prior dropout.
  Validation: 256 fixed samples across all 4 held-out floors and trajectory time.
  Best checkpoints: highest unknown occupied-cell IoU; retain best 3 and last.

These are starting values, not established optimal parameters. Inspect the
first 5 epochs. If occupied IoU/F1 and previews improve, resume each model to
30 total epochs. Plateau and floor errors matter more than GAN loss alone.
30 epochs at batch 2 is about 272,730 generator updates; the SLURM walltime
does not guarantee all 30 epochs will finish in one allocation.

Find each model's last.ckpt under:
  /home/ipa28/nobackup/autodelete/lama_png_scratch/experiments/<run>/models/

Resume the models individually (different checkpoints, same original seed):
  RESUME=/absolute/nosatellite/run/models/last.ckpt EPOCHS=30 \
    sbatch --array=0 train_png_compare.sbatch
  RESUME=/absolute/satellite/run/models/last.ckpt EPOCHS=30 \
    sbatch --array=1 train_png_compare.sbatch

Resume keeps optimizer/training state. Keep batch size, seed, crop, padding,
manifests and satellite settings unchanged when continuing a run. Each
submission creates its own output folder. Last checkpoints save at epoch end,
so an interrupted job can lose the unfinished epoch.

If GPU memory is insufficient, try BATCH_SIZE=1 for BOTH models. Keep float32
initially because the FFC FFT path can be sensitive to mixed precision. Check
actual GPU memory and batches/second in the smoke/pilot before increasing batch.

3. Visualize progress
---------------------
For each experiment:
  png_samples/epoch_XXX/val_000.png ... val_007.png
    Fixed validation cases: GT, observation, simulated footprint, prediction,
    inpainted output, and absolute error in unknown cells.
  samples/
    Existing LaMa training previews every 500 batches.
  validation.jsonl
    One epoch record with raw counts, unknown IoU, F1, precision, recall, MAE.
  metrics/csv/version_0/metrics.csv
    Training losses and validation curves.
  config.yaml
    Exact run configuration, including initialization and input conditions.
  best_checkpoint.txt / last_checkpoint.txt
    Checkpoint paths written after a successful run completes.

TensorBoard logs:
  tensorboard --logdir /home/ipa28/nobackup/autodelete/lama_png_scratch/tb_logs \
    --host 127.0.0.1 --port 6006

Open the TensorBoard port through your usual SSH tunnel. Compare val_unknown_iou
and val_unknown_f1 (higher is better), val_unknown_mae (lower is better), and
the same validation image indices for both models. W&B is disabled; metrics
and images are local files. A TensorBoard logger is actually attached to Trainer.
The included png_lama_prepared/previews are INPUT previews, not model predictions.

4. Compare checkpoints fairly
----------------------------
Use each model's best validation checkpoint. Keep the threshold fixed at 0.5.
First evaluate the FULL validation manifest, not just the 256 monitoring cases:

  export NO_SAT_CKPT=/absolute/nosatellite/run/models/chosen.ckpt
  export SAT_CKPT=/absolute/satellite/run/models/chosen.ckpt
  SPLIT=validate sbatch evaluate_png_compare.sbatch

When model settings/checkpoint selection are settled:
  SPLIT=test sbatch evaluate_png_compare.sbatch

The evaluator scores full native-size maps at batch 1 with no added random
margins (PAD=0) and fixed seeded margins (PAD=64). Dimensions are rounded up
to 32 in both cases for LaMa. It uses the actual observation, scores only
original unknown pixels, and ignores padding in all occupancy metrics.
Outputs under lama_png_scratch/comparisons/<split>_<job>/:
  comparison.json: pooled scores, equal-floor macro IoU, per-floor scores,
    paired satellite-minus-baseline floor deltas and a floor bootstrap interval.
  <model>_pad<N>.csv: one row per observation, including raw confusion counts.
  <model>_pad<N>/*.png: comparison panels.

Undefined precision/recall/IoU is null in final reports. The training checkpoint
monitor uses zero if IoU is undefined. Frames from a floor are correlated;
the floor bootstrap addresses floor variation, not training seed uncertainty.
For a stronger result repeat the paired training with SEED=43 and SEED=44,
keeping these manifests and cached priors fixed. Use seed 42 for final evaluation
of every pair. There are only 4 validation floors; avoid extensive tuning.

Changing padding at evaluation measures padding sensitivity. To measure whether
TRAINING with variable padding helps, also train a matched PAD_MAX=0 pair:
  PAD_MAX=0 EPOCHS=5 sbatch train_png_compare.sbatch
Then compare models on the same evaluation settings. That is a separate
two-by-two ablation (satellite on/off x training padding on/off), requiring
four models. The initial requested run creates just the PAD_MAX=64 pair.

5. What the prepared data contains
---------------------------------
Only train_1 was used to choose training/validation:
  Train:    18,181 pairs, 34 floor/map identities.
  Validate:  2,763 pairs,  4 floor/map identities.
  Test:      8,481 pairs, 19 existing held-out floor/map identities.
No floor ID or decoded GT hash crosses the three sets. A filename parser
removes date, trajectory start position, algorithm and frame index to identify
the floor. Exact GT hashes also link renamed duplicates before splitting.

All 29,425 selected pairs passed dimension and label checks. They contain
57 distinct decoded GT maps. Each GT and its cached prior can be reused in
memory while observations remain indexed directly from the original PNGs.
Both models receive the same sample order, padding, crop and flip schedule
for each seed/epoch. Satellite perturbations use a separate random stream.

Padding is applied jointly to target, actual observation, unknown mask,
valid-region mask and prior. New map/observation pixels are known free (0);
prior padding is unknown (0.5). Native images are placed using the sampled
margins. Extra bottom/right space ensures 512-pixel training crops or
multiple-of-32 validation dimensions. Training tries 8 aligned crop positions
and keeps the one containing the most unknown pixels; it also flips jointly.
Validation uses the full map and fixed per-sample seeded margins.
Artificial padding contributes to neither L1 supervision nor reported
occupancy scores. GAN and feature matching still operate on the padded canvas.

Known observation cells sometimes disagree with GT (3,965,281 cells across all
selected pairs). This pipeline preserves actual observations; it does not
replace them with GT-derived visible pixels. Occupancy input encoding is
0=free, 127/128=unknown, 255=occupied. The prior means building extent, not free
or occupied space. No traversability/valid_space label is required for this
training; valid_region only marks the original PNG rectangle.

Some source PNGs already have occupied edge lines. This setup keeps source
labels intact. Random padding is an edge-position robustness experiment,
not a correction for wrongly labeled source pixels. Audit or repair those
labels separately before a final scientific experiment if they are artifacts.

Synthetic priors are GT-derived closed building envelopes, NOT satellite
imagery. Sixteen plausible scale/rotation/shift/boundary perturbations are
averaged per unique GT. Their settings are recorded in summary/manifests.
This comparison can show whether a simulated extent prior helps LaMa. It
cannot establish the benefit of real satellite inputs, which may be noisier
and have different missing coverage and alignment errors.

Preparation took about 114 seconds locally including prior generation;
cached repeat audit took about 76 seconds. Timing is machine/storage dependent.
57 cached prior PNGs occupy about 191 KB. Padded PNG copies are never written.

6. Rebuild preparation if the source data changes
------------------------------------------------
From the repository root, using a separate output folder:
  python prepare_png_training.py --source "$PNG_DATA_ROOT" \
    --output /home/ipa28/nobackup/autodelete/png_lama_prepared_new \
    --train-splits train_1 --synthetic-priors --workers 8 \
    --uncertainty-samples 16

Set PREPARED to the new output folder before submitting jobs. The source PNGs
remain unchanged. Priors are cached by decoded GT hash plus generation settings.
The preparation script needs only NumPy and Pillow, not a GPU or simulator.

Validation performed locally: full selected-dataset audit, synthetic prior
generation, floor/hash separation, loader alignment/determinism, actual
observation preservation, metric padding exclusion, and input preview review.
PyTorch/CUDA are unavailable locally, so model construction, backward passes,
and SLURM execution still require the supplied cluster smoke jobs.

Repository context:
  https://github.com/ipavery/prediction-aware-exploration/blob/isaac-test/lama/test_train_lama_satellite.sbatch
  https://github.com/ipavery/prediction-aware-exploration/blob/isaac-test/lama/saicinpainting/map_pairs.py
  https://github.com/ipavery/prediction-aware-exploration/blob/isaac-test/lama/saicinpainting/training/trainers/default.py
  https://github.com/ipavery/prediction-aware-exploration/blob/isaac-test/lama/conda_env.yml
  https://github.com/ipavery/prediction-aware-exploration/blob/main/clusters/batch_explore.batch

The runner targets the repository's Lightning 1.2.9 / Hydra 1.1 APIs; it uses
a single GPU without DDP. No existing repository files need patching.
