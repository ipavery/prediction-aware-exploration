#!/usr/bin/env python3
"""Audit PNG pairs, split by floor identity, and cache simulated footprint priors."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'lama'))
from saicinpainting.png_dataset import gray, stable_seed

PATTERN = re.compile(r'^\d{8}_\d{6}_(.+)_-?\d+_-?\d+_[A-Za-z0-9-]+_\d+$')


def scan_one(task):
    root, path, source_split = task
    obs_path = path.parent.parent / 'global_obs' / path.name
    if not obs_path.is_file():
        raise ValueError('Missing observation: ' + str(obs_path))
    match = PATTERN.fullmatch(path.stem)
    if not match:
        raise ValueError('Cannot infer floor ID; unexpected filename: ' + path.name)
    gt, obs = gray(path), gray(obs_path)
    if gt.shape != obs.shape:
        raise ValueError('Pair dimensions differ: ' + str(path))
    if not np.isin(gt, [0, 255]).all() or not np.isin(obs, [0, 127, 128, 255]).all():
        raise ValueError('Unexpected map labels in ' + str(path))
    unknown = (obs == 127) | (obs == 128)
    mismatches = int(((gt != obs) & ~unknown).sum())
    digest = hashlib.sha256(str(gt.shape).encode() + gt.tobytes()).hexdigest()
    return dict(key=path.relative_to(root).as_posix(), gt=path.relative_to(root).as_posix(),
                obs=obs_path.relative_to(root).as_posix(), group=match.group(1), gt_sha256=digest,
                source_split=source_split, height=gt.shape[0], width=gt.shape[1],
                unknown_pixels=int(unknown.sum()), known_disagreements=mismatches)


def assign_splits(rows, fraction, seed):
    # Union floor IDs sharing exact decoded GT, including copies renamed across splits.
    parent = {r['group']: r['group'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_hash = {}
    for row in rows:
        group = row['group']
        other = by_hash.setdefault(row['gt_sha256'], group)
        a, b = sorted((find(group), find(other)))
        parent[b] = a
    heldout = {find(r['group']) for r in rows if r['source_split'] == 'test'}
    candidates = sorted({find(r['group']) for r in rows if r['source_split'] != 'test'} - heldout,
                        key=lambda g: (stable_seed(seed, g), g))
    if len(candidates) < 2:
        raise ValueError('Need at least two independent training floors after removing test overlap')
    nval = max(1, min(len(candidates)-1, round(len(candidates) * fraction)))
    validation = set(candidates[:nval])
    manifests = dict(train=[], validate=[], test=[])
    excluded = Counter()
    for row in rows:
        component = find(row['group'])
        if row['source_split'] == 'test':
            split = 'test'
        elif component in heldout:
            excluded['test_overlap'] += 1
            continue
        elif component in validation:
            split = 'validate'
        else:
            split = 'train'
        if row['unknown_pixels'] == 0:
            excluded[split + '_fully_known'] += 1
            continue
        row['split_group'] = component
        manifests[split].append(row)
    return manifests, dict(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Root with train_1 and test folders')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-splits', nargs='+', default=['train_1'])
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--synthetic-priors', action='store_true')
    parser.add_argument('--uncertainty-samples', type=int, default=16)
    parser.add_argument('--preview-count', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.workers < 1:
        parser.error('validation-fraction must be in (0,1), workers must be positive')
    if 'test' in args.train_splits or len(set(args.train_splits)) != len(args.train_splits):
        parser.error('Training folders must be distinct and cannot include test')
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        parser.error('Output must be outside the input dataset')
    tasks = []
    for split in args.train_splits + ['test']:
        paths = sorted((source / split / 'global_gt').glob('*.png'))
        if not paths:
            parser.error('No GT PNGs in ' + str(source / split / 'global_gt'))
        tasks.extend((source, path, split) for path in paths)
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(scan_one, tasks), 1):
            rows.append(row)
            if i % 2000 == 0:
                print(f'Audited {i}/{len(tasks)} pairs', flush=True)
    manifests, excluded = assign_splits(rows, args.validation_fraction, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    unique = {r['gt_sha256']: r for samples in manifests.values() for r in samples}
    # All labels were decoded and hashed: point repeated GTs to one original file.
    # This enables bounded per-worker caching without duplicating PNGs on disk.
    canonical_gt = {digest: row['gt'] for digest, row in unique.items()}
    for samples in manifests.values():
        for row in samples:
            row['gt'] = canonical_gt[row['gt_sha256']]
    prior_settings = dict(method='closed-envelope', close_radius=3, min_component=64,
                          samples=args.uncertainty_samples, shift_px=3, rotate_deg=2.,
                          scale_jitter=.02, boundary_jitter=1, seed=args.seed)
    if args.synthetic_priors:
        from png_footprint_priors import extract_footprint, uncertain_footprint
        # Settings are part of the cache key so changing parameters cannot reuse stale priors.
        settings_id = hashlib.sha256(json.dumps(prior_settings, sort_keys=True).encode()).hexdigest()[:12]
        prior_dir = output / 'priors' / settings_id
        prior_dir.mkdir(parents=True, exist_ok=True)
        def make_prior(item):
            digest, row = item
            destination = prior_dir / (digest + '.png')
            if not destination.exists():
                walls = gray(source / row['gt']) > 127
                footprint = extract_footprint(walls, close_radius=3, min_component=64)
                rng = np.random.default_rng(stable_seed(args.seed, digest, 'footprint'))
                prior, _ = uncertain_footprint(footprint, rng, samples=args.uncertainty_samples,
                    shift_px=3, rotate_deg=2., scale_jitter=.02, boundary_jitter=1)
                Image.fromarray(np.uint8(np.clip(prior, 0, 1) * 255)).save(destination)
            return digest, destination.relative_to(output).as_posix()
        priors = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (digest, relative) in enumerate(pool.map(make_prior, sorted(unique.items())), 1):
                priors[digest] = relative
                if i % 25 == 0:
                    print(f'Footprint cache: {i}/{len(unique)} unique GTs', flush=True)
        for samples in manifests.values():
            for row in samples:
                row['prior'] = priors[row['gt_sha256']]
    counts = {}
    for split, samples in manifests.items():
        payload = dict(schema_version=1, source_root=str(source), seed=args.seed,
                       prior_kind='synthetic_gt_footprint' if args.synthetic_priors else None,
                       prior_settings=prior_settings if args.synthetic_priors else None, samples=samples)
        (output / (split + '.json')).write_text(json.dumps(payload, indent=2) + '\n')
        counts[split] = dict(pairs=len(samples), groups=len({r['split_group'] for r in samples}),
                            unique_gt=len({r['gt_sha256'] for r in samples}))
    summary = dict(input_pairs=len(rows), selected_training_folders=args.train_splits, counts=counts,
                   excluded=excluded, unique_gt=len(unique),
                   known_disagreements=sum(r['known_disagreements'] for r in rows),
                   maximum_pixels=max(r['height'] * r['width'] for r in rows),
                   preparation_seconds=round(time.perf_counter()-started, 2), prior_settings=prior_settings,
                   warning='Synthetic GT silhouettes measure the utility of an extent prior, not real satellite performance.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.preview_count:
        from saicinpainting.png_dataset import PngMapDataset, save_preview
        for training in (True, False):
            split = 'train' if training else 'validate'
            dataset = PngMapDataset(output / (split + '.json'), use_satellite=args.synthetic_priors, training=training)
            for i in range(min(args.preview_count, len(dataset))):
                batch = dataset[i]
                save_preview(batch, None, output / 'previews' / f'{split}_{i:03d}.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
