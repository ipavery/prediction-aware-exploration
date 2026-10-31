"""CPU regressions for satellite conditioning, without LaMa's optional services.

Run: python -m unittest discover -s tests -p test_lama_satellite.py -v
Requires torch, numpy, opencv, albumentations, Pillow, and OmegaConf. Production
dataset/forward functions are compiled from their AST to avoid initializing the
unrelated Lightning/GAN/perceptual-loss stack. Tensor operations and transforms
use real PyTorch, OpenCV, and Albumentations.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAIRS = load_file('satellite_pairs', 'lama/saicinpainting/map_pairs.py')
WEIGHTS = load_file('satellite_weights', 'lama/saicinpainting/training/trainers/satellite_weights.py')


def extract(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(tree, str(ROOT / path), 'exec'), namespace)
    return namespace


DATA = dict(np=np, cv2=cv2, A=A, Dataset=Dataset,
            paired_map_paths=PAIRS.paired_map_paths, read_map_pair=PAIRS.read_map_pair)
extract('lama/saicinpainting/training/data/datasets.py',
        {'InpaintingTrainActualMaskDataset', 'get_transforms'}, DATA)
EVAL = dict(DATA)
extract('lama/saicinpainting/evaluation/data.py',
        {'InpaintingActualMaskEvalDataset', 'scale_image', 'ceil_modulo', 'pad_img_to_modulo'}, EVAL)
INFER = dict(np=np, cv2=cv2, os=os, default_collate=default_collate,
             move_to_device=lambda batch, device: {k: v.to(device) for k, v in batch.items()},
             load_satellite_image=PAIRS.load_satellite_image)
extract('scripts/lama_pred_utils.py', {'convert_obsimg_to_model_input', 'load_satellite_prior_for_map'}, INFER)


class MiniBase(torch.nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.generator = torch.nn.Conv2d(config.generator.input_nc, 3, 1)


FORWARD = dict(torch=torch, F=F, BaseInpaintingTrainingModule=MiniBase,
               get_ramp=lambda **kw: lambda step: (10, 12),
               make_constant_area_crop_params=lambda **kw: (1, 2, 4, 5))
extract('lama/saicinpainting/training/trainers/default.py',
        {'DefaultInpaintingTrainingModule', 'make_constant_area_crop_batch'}, FORWARD)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('global_gt', 'global_obs', 'global_satellite'):
            (self.root / name).mkdir()
        self.gt = np.zeros((20, 24, 3), dtype=np.uint8)
        self.gt[5:16, 7:20] = 255
        self.obs = self.gt.copy()
        self.obs[6:12, 8:18] = 128
        self.prior = self.gt[:, :, 0]
        for folder, image in (('global_gt', self.gt), ('global_obs', self.obs), ('global_satellite', self.prior)):
            self.assertTrue(cv2.imwrite(str(self.root / folder / 'a.png'), image))

    def dataset(self, satellite=True, transform=None):
        return DATA['InpaintingTrainActualMaskDataset'](str(self.root),
            transform or DATA['get_transforms']('default_map', 16), 1, use_satellite=satellite)

    def test_filename_matching_ignores_extra_observation(self):
        cv2.imwrite(str(self.root / 'global_obs' / '0_unrelated.png'), self.obs)
        dataset = self.dataset()
        self.assertEqual(Path(dataset.obs_files[0]).name, 'a.png')
        self.assertEqual(len(dataset), 1)

    def test_missing_prior_reports_the_matching_filename(self):
        (self.root / 'global_satellite' / 'a.png').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'a.png'):
            self.dataset()

    def test_misaligned_prior_fails_before_augmentation(self):
        cv2.imwrite(str(self.root / 'global_satellite' / 'a.png'), np.zeros((19, 24), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Align it'):
            self.dataset()[0]

    def test_shared_crops_and_flips_keep_prior_aligned(self):
        dataset = self.dataset()
        for _ in range(8):
            sample = dataset[0]
            np.testing.assert_array_equal(sample['image'][0], sample['satellite'][0])
            self.assertEqual(sample['mask'].shape, (1, 16, 16))
            self.assertEqual(sample['satellite'].dtype, np.float32)

    def test_prior_does_not_change_unknown_mask(self):
        transform = DATA['get_transforms']('default_map_val', 16)
        old = self.dataset(False, transform)[0]
        new = self.dataset(True, transform)[0]
        np.testing.assert_array_equal(old['image'], new['image'])
        np.testing.assert_array_equal(old['mask'], new['mask'])
        self.assertNotIn('satellite', old)

    def test_padding_is_unknown_in_prior(self):
        sample = self.dataset(True, DATA['get_transforms']('default_map_val', 32))[0]
        self.assertEqual(sample['satellite'].shape, (1, 32, 32))
        self.assertTrue(np.all(sample['satellite'][0, 0] == 0.5))

    def test_validation_is_deterministic(self):
        dataset = EVAL['InpaintingActualMaskEvalDataset'](str(self.root),
            DATA['get_transforms']('default_map_val', 16), num_frames_to_skip=1, use_satellite=True)
        for key, value in dataset[0].items():
            np.testing.assert_array_equal(value, dataset[0][key])

    def test_eval_scaling_and_modulo_padding_cover_satellite(self):
        dataset = EVAL['InpaintingActualMaskEvalDataset'](str(self.root),
            DATA['get_transforms']('default_map_val', 16), num_frames_to_skip=1,
            use_satellite=True, scale_factor=0.5, pad_out_to_modulo=16)
        sample = dataset[0]
        self.assertEqual(sample['image'].shape, (3, 16, 16))
        self.assertEqual(sample['satellite'].shape, (1, 16, 16))
        self.assertTrue(np.all(sample['satellite'][0, 8:] == 0.5))

    def test_translation_fills_new_coverage_with_unknown(self):
        with mock.patch.object(np.random, 'randint', return_value=np.array([2, -1])):
            shifted = PAIRS.shift_satellite(self.prior.astype(np.float32) / 255, 2)
        self.assertTrue(np.all(shifted[-1] == 0.5))
        self.assertTrue(np.all(shifted[:, :2] == 0.5))
        np.testing.assert_array_equal(shifted[:-1, 2:], self.prior[1:, :-2] / 255)

    def test_deployment_prior_is_loaded_without_gt_and_padded(self):
        # Runtime files are keyed by map ID, not training collection timestamp.
        directory = self.root / 'runtime'
        directory.mkdir()
        cv2.imwrite(str(directory / (self.root.name + '.png')), self.prior)
        prior = INFER['load_satellite_prior_for_map'](str(self.root), (1020, 1024), satellite_dir=str(directory))
        self.assertEqual(prior.shape, (1020, 1024))
        np.testing.assert_array_equal(prior[500:-500, 500:-500], self.prior / 255)
        self.assertEqual(float(prior[0, 0]), 0.5)


class TensorTests(unittest.TestCase):
    def model(self, satellite=True, dropout=0, **kwargs):
        config = SimpleNamespace(generator=SimpleNamespace(input_nc=5 if satellite else 4))
        return FORWARD['DefaultInpaintingTrainingModule'](config,
            condition_on_satellite=satellite, satellite_dropout=dropout, **kwargs)

    def batch(self):
        return dict(image=torch.rand(2, 3, 8, 9), mask=torch.ones(2, 1, 8, 9),
                    satellite=torch.ones(2, 1, 8, 9))

    def test_prior_survives_inside_unknown_regions(self):
        model = self.model().eval()
        captured = []
        model.generator.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        output = model(self.batch())
        self.assertEqual(tuple(captured[0].shape), (2, 5, 8, 9))
        self.assertEqual(float(captured[0][:, :3].abs().sum()), 0)
        self.assertTrue(torch.all(captured[0][:, 3:] == 1))
        self.assertEqual(tuple(output['predicted_image'].shape), (2, 3, 8, 9))

    def test_satellite_channel_receives_gradient(self):
        model = self.model()
        output = model(self.batch())
        output['predicted_image'].sum().backward()
        self.assertGreater(float(model.generator.weight.grad[:, 4].abs().sum()), 0)

    def test_dropout_replaces_entire_prior_only_during_training(self):
        model = self.model(dropout=1)
        captured = []
        model.generator.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        model.train()(self.batch())
        model.eval()(self.batch())
        self.assertTrue(torch.all(captured[0][:, 4] == 0.5))
        self.assertTrue(torch.all(captured[1][:, 4] == 1))

    def test_known_observations_are_preserved(self):
        batch = self.batch()
        batch['mask'][:, :, :3] = 0
        output = self.model().eval()(batch)
        self.assertTrue(torch.allclose(output['inpainted'][:, :, :3], batch['image'][:, :, :3]))

    def test_legacy_model_still_receives_four_channels(self):
        model = self.model(False).eval()
        captured = []
        model.generator.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        model(self.batch())
        self.assertEqual(captured[0].shape[1], 4)

    def test_missing_and_wrong_shape_satellite_are_explicit_errors(self):
        model = self.model()
        batch = self.batch()
        del batch['satellite']
        with self.assertRaisesRegex(ValueError, 'requires'):
            model(batch)
        batch['satellite'] = torch.ones(2, 1, 7, 9)
        with self.assertRaisesRegex(ValueError, 'shape'):
            model(batch)

    def test_internal_crop_and_resize_also_transform_prior(self):
        model = self.model(const_area_crop_kwargs={})
        self.assertEqual(tuple(model(self.batch())['satellite'].shape), (2, 1, 4, 5))
        model = self.model(rescale_scheduler_kwargs={})
        model.global_step = 0
        self.assertEqual(tuple(model(self.batch())['satellite'].shape), (2, 1, 10, 12))

    def test_old_weights_preserve_initial_predictions_after_expansion(self):
        old = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 3, padding=1))
        new = torch.nn.Sequential(torch.nn.Conv2d(5, 3, 3, padding=1))
        adapted = WEIGHTS.adapt_satellite_generator_weights(new, old.state_dict(), True)
        new.load_state_dict(adapted)
        image = torch.randn(2, 4, 10, 12)
        prior = torch.randn(2, 1, 10, 12)
        self.assertTrue(torch.allclose(new(torch.cat([image, prior], dim=1)), old(image), atol=1e-6))
        self.assertEqual(float(new[0].weight[:, 4].abs().sum()), 0)

    def test_other_architecture_mismatches_are_rejected(self):
        old = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3))
        new = torch.nn.Sequential(torch.nn.Conv2d(5, 3, 3))
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            WEIGHTS.adapt_satellite_generator_weights(new, old.state_dict(), True)

    def test_warm_start_loads_generator_without_overwriting_other_state(self):
        model = self.model()
        old = torch.nn.Conv2d(4, 3, 1)
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'old.ckpt')
            torch.save({'state_dict': {'generator.' + k: v for k, v in old.state_dict().items()}}, path)
            WEIGHTS.load_pretrained_generator(model, path)
        self.assertTrue(torch.allclose(model.generator.weight[:, :4], old.weight))
        self.assertEqual(float(model.generator.weight[:, 4].abs().sum()), 0)

    def test_inference_builds_neutral_or_supplied_prior_with_shared_padding(self):
        transform = DATA['get_transforms']('default_map_eval')
        obs = np.full((17, 19, 3), 0.5, np.float32)
        neutral, mask = INFER['convert_obsimg_to_model_input'](obs, transform, 'cpu', use_satellite=True)
        self.assertEqual(tuple(neutral['satellite'].shape), (1, 1, 32, 32))
        self.assertTrue(torch.all(neutral['satellite'] == 0.5))
        batch, _ = INFER['convert_obsimg_to_model_input'](obs, transform, 'cpu', satellite=np.ones((17, 19), np.float32))
        self.assertEqual(batch['image'].shape[2:], batch['satellite'].shape[2:])
        self.assertGreater(int((batch['satellite'] == 1).sum()), 0)
        self.assertEqual(float(batch['satellite'][0, 0, 0, 0]), 0.5)


class ConfigTests(unittest.TestCase):
    def test_satellite_config_is_opt_in(self):
        old = OmegaConf.load(str(ROOT / 'lama/configs/training/big-lama.yaml'))
        new = OmegaConf.load(str(ROOT / 'lama/configs/training/big-lama-satellite.yaml'))
        self.assertEqual(old.generator.input_nc, 4)
        self.assertFalse(old.training_model.get('condition_on_satellite', False))
        self.assertEqual(new.generator.input_nc, 5)
        self.assertEqual(new.generator.output_nc, 3)
        self.assertTrue(new.training_model.condition_on_satellite)

    def test_data_config_uses_all_pairs_and_deterministic_validation(self):
        cfg = OmegaConf.load(str(ROOT / 'lama/configs/training/data/satellite-maps.yaml'))
        self.assertEqual(cfg.train.num_frames_to_skip, 1)
        self.assertTrue(cfg.train.use_satellite)
        self.assertEqual(cfg.val.transform_variant, 'default_map_val')
        self.assertTrue(cfg.val.use_satellite)


if __name__ == '__main__':
    unittest.main()
