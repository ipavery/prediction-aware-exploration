#!/usr/bin/env python3
"""Paired full-map evaluation; scores only unknown cells and groups by floor."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

from saicinpainting.png_dataset import PngMapDataset, unknown_metrics, scores, save_preview
from saicinpainting.training.trainers import make_training_model


def add_counts(total, counts):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--model', nargs=2, action='append', metavar=('LABEL', 'CHECKPOINT'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--padding-max', nargs='+', type=int, default=[0, 64])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--preview-count', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        parser.error('Model labels must be unique')
    report = {}
    floor_scores = {}
    for label, checkpoint_path in args.model:
        checkpoint_path = Path(checkpoint_path)
        config = OmegaConf.load(checkpoint_path.parent.parent / 'config.yaml')
        with open_dict(config.training_model):
            config.training_model.predict_only = True
        model = make_training_model(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        weights = {k[len('generator.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('generator.')}
        model.generator.load_state_dict(weights, strict=True)
        del checkpoint, weights
        model.to(args.device).eval()
        conditioned = bool(config.training_model.get('condition_on_satellite', False))
        for padding in args.padding_max:
            dataset = PngMapDataset(args.manifest, root=args.data_root, use_satellite=conditioned,
                                   padding_max=padding, seed=args.seed, limit=args.limit)
            pooled, per_floor, records = {}, {}, []
            with torch.inference_mode():
                for i in range(len(dataset)):
                    sample = dataset[i]
                    batch = {k: torch.from_numpy(v[None]).to(args.device) for k, v in sample.items()
                             if k in ('mask', 'satellite')}
                    # Inference sees observations; true GT remains outside the model.
                    batch['image'] = torch.from_numpy(sample['observation'][None]).to(args.device)
                    prediction = model(batch)['predicted_image'][0].float().mean(0).cpu().numpy()
                    counts = unknown_metrics(prediction, sample['image'][0], sample['evaluation_mask'][0])
                    row = dataset.rows[i]
                    add_counts(pooled, counts)
                    add_counts(per_floor.setdefault(row['split_group'], {}), counts)
                    records.append(dict(key=row['key'], floor=row['split_group'], **counts, **scores(counts)))
                    if i < args.preview_count:
                        save_preview(sample, prediction, args.output / f'{label}_pad{padding}' / f'{i:03d}.png')
                    if (i + 1) % 250 == 0:
                        print(f'{label}, pad={padding}: {i+1}/{len(dataset)}', flush=True)
            name = f'{label}_pad{padding}'
            by_floor = {floor: scores(counts) for floor, counts in per_floor.items()}
            valid_iou = [v['occupied_iou'] for v in by_floor.values() if v['occupied_iou'] is not None]
            report[name] = dict(padding_policy='gray_observation_black_footprint_v2',
                                pooled=scores(pooled), counts=pooled,
                                floor_macro_iou=statistics.mean(valid_iou) if valid_iou else None,
                                floors=by_floor, samples=len(dataset), checkpoint=str(checkpoint_path))
            floor_scores[name] = {floor: v['occupied_iou'] for floor, v in by_floor.items()}
            with (args.output / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
            print(name, json.dumps(report[name]['pooled']), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Paired floor deltas, rather than treating thousands of related frames as independent maps.
    if len(labels) == 2:
        deltas = {}
        for padding in args.padding_max:
            a, b = (floor_scores[f'{label}_pad{padding}'] for label in labels)
            differences = [b[f] - a[f] for f in sorted(a.keys() & b.keys())
                           if a[f] is not None and b[f] is not None]
            if differences:
                rng = np.random.RandomState(args.seed)
                bootstrap = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
                deltas[f'pad{padding}'] = dict(direction=f'{labels[1]} minus {labels[0]}',
                    mean_floor_iou_delta=float(np.mean(differences)), floors=len(differences),
                    bootstrap_95_percent_interval=np.percentile(bootstrap, [2.5, 97.5]).tolist())
        report['paired_floor_comparison'] = deltas
        (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
