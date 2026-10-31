import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import yaml
from torch.utils.data._utils.collate import default_collate

import sys
sys.path.append('../lama')
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.training.data.datasets import get_transforms
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.map_pairs import load_satellite_image

def get_lama_transform(transform_variant, out_size):
    map_transform = get_transforms(transform_variant, out_size)
    return map_transform

def convert_obsimg_to_model_input(obs_img, map_transform, device, satellite=None, use_satellite=False):
    inputs = dict(image=obs_img, obs_img=obs_img)
    if use_satellite or satellite is not None:
        if satellite is None:
            satellite = np.full(obs_img.shape[:2], 0.5, dtype=np.float32)
        if tuple(satellite.shape) != tuple(obs_img.shape[:2]):
            raise ValueError('Satellite and observed map must use the same grid')
        satellite = np.asarray(satellite, dtype=np.float32)
        if not np.isfinite(satellite).all() or satellite.min() < 0 or satellite.max() > 1:
            raise ValueError('Satellite must be normalized to [0, 1] before inference')
        inputs['satellite'] = satellite
    transformed = map_transform(**inputs)
    obs_img = np.transpose(transformed['obs_img'], (2, 0, 1))
    mask = ((obs_img[0] > 0.49) & (obs_img[0] < 0.51)).astype(np.float32)[None, ...]
    sample = {'image': obs_img, 'mask': mask}
    if 'satellite' in inputs:
        sample['satellite'] = np.ascontiguousarray(transformed['satellite'][None, ...], dtype=np.float32)
    input_batch = move_to_device(default_collate([sample]), device)
    return input_batch, mask


def load_satellite_prior_for_map(map_folder_path, padded_shape, satellite_dir=None, padding=500):
    map_id = os.path.basename(os.path.normpath(map_folder_path))
    if satellite_dir is not None:
        path = os.path.join(satellite_dir, map_id + '.png')
        if not os.path.isfile(path):
            raise FileNotFoundError('Missing supplied satellite footprint: ' + path)
    else:
        path = os.path.join(map_folder_path, 'satellite.png')
        if not os.path.isfile(path):
            return None
    interior_shape = (padded_shape[0] - 2 * padding, padded_shape[1] - 2 * padding)
    prior = load_satellite_image(path, interior_shape)
    return np.pad(prior, padding, mode='constant', constant_values=0.5) if padding else prior


# Visualization
def load_lama_model(model_path, checkpoint_name='best.ckpt', device='cuda'):
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    checkpoint_path = os.path.join(model_path, 'models', checkpoint_name)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device).to(device)
    model.freeze()
    return model


def visualize_prediction(batch_pred, mask):
    cur_gt = batch_pred['image'][0].permute(1, 2, 0).cpu().numpy()
    cur_res = np.clip(batch_pred['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cur_gt = np.clip(cur_gt * 255, 0, 255).astype('uint8')
    cur_gt = cv2.cvtColor(cur_gt, cv2.COLOR_RGB2BGR)

    gt_masked = cur_gt.copy()
    gt_masked[mask[0] > 0] = 122
    disp_output = np.vstack([cur_gt, gt_masked, cur_res])
    cv2.imwrite('lama_pred.png', disp_output)
    return cur_res