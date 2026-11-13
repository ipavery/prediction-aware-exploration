"""PNG pairs with online, aligned padding. No simulator or OpenCV dependency."""
import hashlib
import json
import multiprocessing
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')).copy()


@lru_cache(maxsize=128)
def cached_gray(path):
    image = gray(path)
    image.flags.writeable = False
    return image


def stratified_rows(rows, limit):
    if not limit or limit >= len(rows):
        return rows
    groups = {}
    for row in rows:
        groups.setdefault(row['split_group'], []).append(row)
    ordered = sorted(groups)
    chosen = []
    # Spread monitoring samples over floors and trajectory time, rather than taking one prefix.
    for i, group in enumerate(ordered):
        budget = limit // len(ordered) + int(i < limit % len(ordered))
        samples = groups[group]
        chosen.extend(samples[j] for j in np.linspace(0, len(samples)-1, min(budget, len(samples)), dtype=int))
    return chosen


def unknown_metrics(prediction, target, unknown):
    """Counts on the supplied evaluation mask, excluding artificial padding."""
    selected = np.asarray(unknown, dtype=bool)
    pred = np.asarray(prediction)[selected]
    truth = np.asarray(target)[selected]
    p, t = pred >= .5, truth >= .5
    return dict(tp=int((p & t).sum()), fp=int((p & ~t).sum()),
                fn=int((~p & t).sum()), tn=int((~p & ~t).sum()),
                absolute_error=float(np.abs(pred - truth).sum()), pixels=int(selected.sum()))


def scores(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    def ratio(a, b):
        return a / b if b else None
    return dict(occupied_iou=ratio(tp, tp + fp + fn),
                occupied_f1=ratio(2 * tp, 2 * tp + fp + fn),
                occupied_precision=ratio(tp, tp + fp), occupied_recall=ratio(tp, tp + fn),
                unknown_mae=ratio(counts['absolute_error'], counts['pixels']),
                unknown_pixels=counts['pixels'])


class PngMapDataset:
    def __init__(self, manifest, root=None, use_satellite=False, training=False,
                 crop_size=512, padding_max=64, seed=42, satellite_shift_px=0,
                 satellite_dropout=0., limit=None):
        self.manifest = Path(manifest)
        payload = json.loads(self.manifest.read_text())
        self.root = Path(root or payload['source_root'])
        self.rows = stratified_rows(payload['samples'], limit)
        self.use_satellite, self.training = use_satellite, training
        self.crop_size, self.padding_max, self.seed = int(crop_size), int(padding_max), int(seed)
        self.satellite_shift_px, self.satellite_dropout = int(satellite_shift_px), float(satellite_dropout)
        if self.crop_size < 32 or self.crop_size % 32 or self.padding_max < 0:
            raise ValueError('crop_size must be a positive multiple of 32 and padding_max nonnegative')
        if not 0 <= self.satellite_dropout <= 1 or self.satellite_shift_px < 0:
            raise ValueError('Invalid satellite augmentation settings')
        if not self.rows:
            raise ValueError('Empty manifest: ' + str(manifest))
        if use_satellite and any(not row.get('prior') for row in self.rows):
            raise ValueError('Satellite model needs a prior for EVERY manifest sample')
        # Shared epoch reaches existing workers; padding/crops do not depend on model RNG.
        self.epoch = multiprocessing.Value('i', 0)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch.value = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = np.random.RandomState(stable_seed(self.seed, self.epoch.value if self.training else 0, row['key']))
        gt = cached_gray(self.root / row['gt']).astype(np.float32) / 255.
        obs = gray(self.root / row['obs'])
        unknown = (obs == 128) | (obs == 127)
        height, width = gt.shape
        top, bottom, left, right = rng.randint(0, self.padding_max + 1, size=4)
        hh, ww = height + top + bottom, width + left + right
        if self.training:
            hh, ww = max(hh, self.crop_size), max(ww, self.crop_size)
        else:
            hh, ww = ((hh + 31) // 32) * 32, ((ww + 31) // 32) * 32
        image = np.zeros((hh, ww), np.float32)
        observed = np.full((hh, ww), .5, np.float32)
        mask = np.ones((hh, ww), np.float32)  # Unobserved exterior is unknown to the model.
        valid = np.zeros((hh, ww), np.float32)
        region = np.s_[top:top + height, left:left + width]
        image[region], mask[region], valid[region] = gt, unknown, 1
        observed[region] = obs.astype(np.float32) / 255.
        arrays = dict(image=image, observation=observed, mask=mask, valid_region=valid)
        if self.use_satellite:
            prior = cached_gray(self.manifest.parent / row['prior']).astype(np.float32) / 255.
            if prior.shape != gt.shape:
                raise ValueError('Prior geometry differs from GT: ' + row['key'])
            # Source PNG contains the whole building: added exterior is outside its footprint.
            canvas = np.zeros((hh, ww), np.float32)
            canvas[region] = prior
            arrays['satellite'] = canvas
        if self.training:
            # Try several native-scale crops, retaining one with prediction targets.
            # This avoids training primarily on empty exterior on long, narrow maps.
            candidates = [(int(rng.randint(hh - self.crop_size + 1)),
                           int(rng.randint(ww - self.crop_size + 1))) for _ in range(8)]
            y, x = max(candidates, key=lambda p: (
                mask[p[0]:p[0]+self.crop_size, p[1]:p[1]+self.crop_size] *
                valid[p[0]:p[0]+self.crop_size, p[1]:p[1]+self.crop_size]).sum())
            arrays = {k: v[y:y+self.crop_size, x:x+self.crop_size] for k, v in arrays.items()}
            flip_y, flip_x = rng.randint(0, 2, size=2)
            arrays = {k: v[::(-1 if flip_y else 1), ::(-1 if flip_x else 1)] for k, v in arrays.items()}
            # Sample prior-only perturbations AFTER shared geometry to keep paired runs identical.
            if self.use_satellite:
                prior_rng = np.random.RandomState(stable_seed(self.seed, self.epoch.value, row['key'], 'prior'))
                prior = arrays['satellite']
                if self.satellite_shift_px:
                    dy, dx = prior_rng.randint(-self.satellite_shift_px, self.satellite_shift_px + 1, size=2)
                    shifted = np.zeros_like(prior)
                    h, w = prior.shape
                    sy, sx = max(0, -dy), max(0, -dx)
                    ty, tx = max(0, dy), max(0, dx)
                    sh, sw = h - abs(dy), w - abs(dx)
                    if sh > 0 and sw > 0:
                        shifted[ty:ty+sh, tx:tx+sw] = prior[sy:sy+sh, sx:sx+sw]
                    prior = shifted
                if prior_rng.rand() < self.satellite_dropout:
                    prior = np.full_like(prior, .5)
                arrays['satellite'] = prior
        # Input mask includes unknown padding; evaluation/supervision does not.
        arrays['evaluation_mask'] = arrays['mask'] * arrays['valid_region']
        for key in ('image', 'observation'):
            arrays[key] = np.repeat(arrays[key][None], 3, axis=0)
        return {k: np.ascontiguousarray(v if k in ('image', 'observation') else v[None]) for k, v in arrays.items()}


def save_preview(batch, prediction, output):
    """Six labeled panels, accepting NumPy CHW arrays."""
    gt, mask = batch['image'][0], batch['mask'][0]
    obs = batch.get('observation', batch['image'])[0] * (1 - mask) + .5 * mask
    prior = batch.get('satellite', np.full_like(batch['mask'], .5))[0]
    panels = [('GT', gt), ('Observation', obs), ('Footprint prior (simulated)', prior)]
    if prediction is not None:
        prediction = np.clip(prediction, 0, 1)
        if prediction.ndim == 3:
            prediction = prediction.mean(axis=0)
        blended = obs * (1 - mask) + prediction * mask
        evaluation_mask = batch.get('evaluation_mask', batch['mask'])[0]
        error = np.abs(prediction - gt) * evaluation_mask
        panels += [('Prediction', prediction), ('Inpainted', blended), ('Unknown error', error)]
    tiles = []
    for label, a in panels:
        im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).convert('RGB')
        im.thumbnail((360, 360))
        tile = Image.new('RGB', (360, im.height + 28), '#dddddd')
        tile.paste(im, (0, 28))
        ImageDraw.Draw(tile).text((8, 8), label, fill='black')
        tiles.append(tile)
    tile_height = tiles[0].height
    canvas = Image.new('RGB', (1080, ((len(tiles) + 2) // 3) * tile_height), '#dddddd')
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 3) * 360, (i // 3) * tile_height))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
