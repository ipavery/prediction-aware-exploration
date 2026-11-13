"""Run with: python -m unittest test_png_training -v (NumPy + Pillow only)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'lama'))
from saicinpainting.png_dataset import PngMapDataset, unknown_metrics, scores, stratified_rows
from prepare_png_training import assign_splits


class PngTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        gt = np.zeros((81, 123), np.uint8)
        gt[15:70, 20] = 255
        gt[25, 20:100] = 255
        obs = gt.copy()
        obs[:, 30:] = 128
        obs[10, 10] = 255  # A real observed discrepancy must reach the model.
        prior = np.zeros_like(gt)
        prior[10:75, 15:105] = 255
        for name, array in [('gt', gt), ('obs', obs), ('prior', prior)]:
            Image.fromarray(array).save(self.root / (name + '.png'))
        row = dict(key='sample', gt='gt.png', obs='obs.png', prior='prior.png', split_group='floor')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(source_root=str(self.root), samples=[row])))

    def dataset(self, **kwargs):
        return PngMapDataset(self.manifest, **kwargs)

    def test_padding_semantics(self):
        sample = self.dataset(use_satellite=True)[0]
        exterior = sample['valid_region'] == 0
        self.assertTrue(exterior.any())
        self.assertTrue((sample['mask'][exterior] == 0).all())
        self.assertTrue((sample['image'][0:1][exterior] == 0).all())
        self.assertTrue((sample['satellite'][exterior] == .5).all())
        self.assertEqual(sample['image'].shape[1] % 32, 0)
        self.assertEqual(sample['image'].shape[2] % 32, 0)

    def test_paired_geometry_and_epoch_changes(self):
        a = self.dataset(training=True, crop_size=256)
        b = self.dataset(training=True, crop_size=256, use_satellite=True,
                         satellite_shift_px=2, satellite_dropout=.1)
        before = a[0]
        for epoch in (0, 1, 7):
            a.set_epoch(epoch)
            b.set_epoch(epoch)
            for key in ('image', 'observation', 'mask', 'valid_region'):
                np.testing.assert_array_equal(a[0][key], b[0][key])
            self.assertEqual(a[0]['image'].shape, (3, 256, 256))
            self.assertEqual(set(np.unique(a[0]['image'])), {0., 1.})
            self.assertTrue(all(v.flags.c_contiguous for v in b[0].values()))
        self.assertFalse(np.array_equal(before['image'], a[0]['image']))

    def test_validation_deterministic(self):
        dataset = self.dataset(use_satellite=True)
        first = dataset[0]
        dataset.set_epoch(100)
        for key in first:
            np.testing.assert_array_equal(first[key], dataset[0][key])

    def test_actual_observation_preserved(self):
        sample = self.dataset()[0]
        known = (sample['mask'][0] == 0) & (sample['valid_region'][0] == 1)
        self.assertEqual(int(((sample['observation'][0] != sample['image'][0]) & known).sum()), 1)

    def test_padding_excluded_from_metrics(self):
        truth = np.array([[1., 0.], [1., 0.]])
        prediction = np.array([[.8, .7], [.2, .1]])
        unknown = np.ones_like(truth)
        counts = unknown_metrics(prediction, truth, unknown)
        padded = unknown_metrics(np.pad(prediction, 30, constant_values=1),
                                 np.pad(truth, 30), np.pad(unknown, 30))
        self.assertEqual(counts, padded)
        self.assertAlmostEqual(scores(counts)['occupied_iou'], 1/3)
        self.assertAlmostEqual(scores(counts)['occupied_f1'], .5)
        self.assertIsNone(scores(unknown_metrics(truth, truth, np.zeros_like(truth)))['occupied_iou'])

    def test_duplicate_gt_cannot_leak(self):
        rows = [dict(group=g, gt_sha256=h, source_split=s, unknown_pixels=10)
                for g, h, s in [('a', 'same', 'train_1'), ('b', 'same', 'test'),
                                ('c', 'c', 'train_1'), ('d', 'd', 'train_1'),
                                ('e', 'd', 'train_1')]]
        split, excluded = assign_splits(rows, .1, 42)
        self.assertEqual(excluded['test_overlap'], 1)
        seen = [{r['gt_sha256'] for r in samples} for samples in split.values()]
        self.assertFalse(seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])

    def test_monitor_spans_floors_and_time(self):
        rows = [dict(split_group=str(g), frame=i) for g in range(4) for i in range(100)]
        subset = stratified_rows(rows, 16)
        self.assertEqual(len(subset), 16)
        self.assertEqual({r['split_group'] for r in subset}, {'0', '1', '2', '3'})
        self.assertEqual({r['frame'] for r in subset}, {0, 33, 66, 99})


if __name__ == '__main__':
    unittest.main()
