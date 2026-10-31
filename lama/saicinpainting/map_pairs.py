"""Matched occupancy/observation/footprint samples; footprint is an extent prior."""
import glob
import os

import cv2
import numpy as np


SATELLITE_UNKNOWN = 0.5


def paired_map_paths(root, stride=1, use_satellite=False):
    if stride < 1:
        raise ValueError('num_frames_to_skip must be at least 1')
    gt_paths = sorted(glob.glob(os.path.join(root, 'global_gt', '*.png')))[::stride]
    obs_paths = [os.path.join(root, 'global_obs', os.path.basename(p)) for p in gt_paths]
    sat_paths = [os.path.join(root, 'global_satellite', os.path.basename(p)) for p in gt_paths]
    for paths in (obs_paths, sat_paths if use_satellite else []):
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError('Missing matching map image: ' + path)
    return gt_paths, obs_paths, sat_paths


def load_satellite_image(path, expected_shape=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Cannot read satellite footprint: ' + str(path))
    if expected_shape is not None and tuple(image.shape) != tuple(expected_shape):
        raise ValueError('Satellite footprint {} has shape {}; expected {}. Align it to the map grid first.'.format(
            path, image.shape, expected_shape))
    return image.astype(np.float32) / 255.0


def shift_satellite(prior, maximum_pixels):
    if maximum_pixels < 0:
        raise ValueError('satellite_shift_px must be nonnegative')
    if not maximum_pixels:
        return prior
    dx, dy = np.random.randint(-maximum_pixels, maximum_pixels + 1, size=2)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(prior, matrix, (prior.shape[1], prior.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=SATELLITE_UNKNOWN)


def _float_image(image):
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def read_map_pair(gt_path, obs_path, transform, satellite_path=None, satellite_shift_px=0):
    gt = cv2.imread(gt_path)
    obs = cv2.imread(obs_path)
    if gt is None or obs is None:
        raise ValueError('Cannot read GT/observation pair: {} / {}'.format(gt_path, obs_path))
    if gt.shape != obs.shape:
        raise ValueError('GT/observation dimensions differ: {} / {}'.format(gt_path, obs_path))
    inputs = dict(image=cv2.cvtColor(gt, cv2.COLOR_BGR2RGB),
                  obs_img=cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
    if satellite_path is not None:
        prior = load_satellite_image(satellite_path, gt.shape[:2])
        inputs['satellite'] = shift_satellite(prior, satellite_shift_px)
    transformed = transform(**inputs)
    image = _float_image(transformed['image']).transpose(2, 0, 1)
    observed = _float_image(transformed['obs_img']).transpose(2, 0, 1)
    mask = ((observed[0] > 0.49) & (observed[0] < 0.51)).astype(np.float32)[None, ...]
    result = dict(image=np.ascontiguousarray(image), mask=np.ascontiguousarray(mask))
    if satellite_path is not None:
        prior = transformed['satellite'].astype(np.float32)
        if prior.shape != image.shape[1:]:
            raise ValueError('The satellite transform must preserve alignment with the map')
        result['satellite'] = np.ascontiguousarray(prior[None, ...])
    return result
