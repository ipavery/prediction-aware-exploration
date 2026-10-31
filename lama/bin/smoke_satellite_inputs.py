#!/usr/bin/env python3
"""Check a real five-channel LaMa forward pass in the existing pipe environment.

Run from lama/: python bin/smoke_satellite_inputs.py --data-dir /path/to/train_3
Optionally add --checkpoint /path/to/old/best.ckpt --device cuda:0.
Does not train, write model weights, or change input data.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import InpaintingTrainActualMaskDataset, get_transforms
from saicinpainting.training.trainers import make_training_model
from saicinpainting.training.trainers.satellite_weights import load_pretrained_generator
from saicinpainting.evaluation.utils import move_to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / 'configs/training/big-lama-satellite.yaml'
    with config_path.open() as stream:
        config = OmegaConf.create(yaml.safe_load(stream))
    config.trainer = {'kwargs': {}}
    config.training_model.predict_only = True
    model = make_training_model(config)
    if args.checkpoint:
        load_pretrained_generator(model, args.checkpoint)
    model = model.to(args.device).eval()
    dataset = InpaintingTrainActualMaskDataset(args.data_dir,
        get_transforms('default_map_val', 256), 1, use_satellite=True)
    if not len(dataset):
        raise ValueError('No paired samples found')
    batch = move_to_device(default_collate([dataset[0]]), args.device)
    with torch.no_grad():
        output = model(batch)
    assert tuple(output['predicted_image'].shape) == (1, 3, 256, 256)
    known = (1 - batch['mask']).expand_as(batch['image'])
    assert torch.allclose(output['inpainted'] * known, batch['image'] * known)
    print('PASS: [1, 5, 256, 256] input -> [1, 3, 256, 256] occupancy output')
    print('Known map cells preserved; prior supplied through the separate fifth channel')


if __name__ == '__main__':
    main()
