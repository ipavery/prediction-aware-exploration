"""Regression tests for collection exits, without GPU or simulator dependencies.

Run from the repository root: python -m unittest discover -s tests -v
Needs NumPy and Pillow. Production functions are compiled from explore.py's AST
so their control flow is exercised with deterministic mapper/planner stand-ins.
Image writes/readback use Pillow in place of OpenCV's PNG codec.
"""
import ast
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts' / 'explore.py'
FUNCTIONS = {
    'dummy_context_manager', 'update_mission_status',
    'is_locked_frontier_center_valid', 'reselect_frontier_from_frontier_region_centers',
    'save_lama_training_pair', 'determine_local_planner',
    'run_exploration_for_map', 'determine_use_model',
}


def write_png(path, array):
    Image.fromarray(np.rint(array).astype(np.uint8)).save(str(path))
    return True


def read_png(path):
    try:
        with Image.open(path) as image:
            return np.array(image.convert('RGB'))
    except (OSError, ValueError):
        return None


def load_functions(namespace):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    exec(compile(tree, str(SOURCE), 'exec'), namespace)


class CollectionExitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_root = self.temp.name
        self.gt = np.zeros((1008, 1009), dtype=float)
        self.gt[500, 500] = 1
        self.start = np.array([503, 503])
        self.frontiers = np.array([[505, 505]])
        self.path_available = True
        self.frontier_exception_after_moves = None
        self.frontier_calls = 0
        self.empty_on_refresh = False
        self.initial_observation_fails = False
        self.astar_goals = []
        self.mapper = None
        test = self

        class Mapper:
            def __init__(self, gt_map, *args, **kwargs):
                self.gt_map = gt_map.copy()
                self.obs_map = np.full_like(gt_map, 0.5)
                self.observations = 0
                test.mapper = self

            def observe_and_accumulate_given_pose(self, pose):
                if test.initial_observation_fails and self.observations == 0:
                    raise RuntimeError('initial lidar failed')
                self.obs_map[tuple(pose)] = self.gt_map[tuple(pose)]
                self.observations += 1

            def get_inflated_planning_maps(self, **kwargs):
                return np.ones_like(self.gt_map, dtype=np.float32)

        class Planner:
            def __init__(self, **kwargs):
                pass

            def get_frontier_centers_given_obs_map(self, obs):
                test.frontier_calls += 1
                # The initial checkpoint must already exist before planning.
                test.assertTrue(test.pair_path('global_obs').is_file())
                if (test.frontier_exception_after_moves is not None and
                        test.mapper.observations - 1 >= test.frontier_exception_after_moves):
                    raise RuntimeError('simulated frontier detection failure')
                candidates = test.frontiers
                if test.empty_on_refresh and test.frontier_calls > 1:
                    candidates = np.empty((0, 2), dtype=int)
                return candidates, None, len(candidates)

            def score_frontiers(self, candidates, *args, **kwargs):
                return np.asarray(candidates), np.arange(len(candidates)), None, None, 0, 0

        def astar(weights, start, goal, **kwargs):
            test.assertIsNotNone(goal, 'A* must never receive an empty frontier goal')
            test.astar_goals.append(np.array(goal))
            if not test.path_available:
                return None
            return np.array([start, goal])

        self.pyplot = SimpleNamespace(subplots=mock.Mock(side_effect=AssertionError('plotting is disabled')),
                                      close=mock.Mock())
        self.namespace = dict(np=np, os=os, time=time, json=json, traceback=traceback,
            datetime=datetime, contextmanager=contextmanager, plt=self.pyplot,
            cv2=SimpleNamespace(imwrite=write_png),
            smu=SimpleNamespace(convert_01_single_channel_to_0_255_3_channel=lambda a: np.repeat(a[:, :, None], 3, axis=2) * 255),
            sim_utils=SimpleNamespace(Mapper=Mapper, FrontierPlanner=Planner,
                psuedo_traj_controller=lambda x, y, **kw: np.array([x[-1], y[-1]])),
            pyastar2d=SimpleNamespace(astar_path=astar),
            collect_opts=SimpleNamespace(log_iou=False, cur_pose_dist_threshold_m=1,
                show_plt_freq=50, mission_time=200),
            get_padded_obs_map=lambda a: a, get_padded_gt_map=lambda a: a)
        load_functions(self.namespace)

    def pair_path(self, folder):
        return Path(self.data_root) / 'train_3' / folder / 'sample.png'

    def run_collection(self, max_steps=1, target_known_fraction=None):
        self.output = io.StringIO()
        with redirect_stdout(self.output):
            self.namespace['run_exploration_for_map'](
                self.gt, '20261026_120000_test_503_503', [], None, None, {},
                {'pixel_per_meter': 1}, 'nearest', self.start, True, True,
                max_steps=max_steps, target_known_fraction=target_known_fraction,
                training_data_root=self.data_root, sample_name='sample',
                save_experiment_artifacts=False)

    def assert_pair(self, observed_cells):
        gt = read_png(self.pair_path('global_gt'))
        obs = read_png(self.pair_path('global_obs'))
        self.assertIsNotNone(gt)
        self.assertIsNotNone(obs)
        self.assertEqual(gt.shape, (8, 9, 3))
        self.assertEqual(obs.shape, gt.shape)
        self.assertEqual(np.count_nonzero(obs[:, :, 0] != 128), observed_cells)
        self.assertFalse(list(Path(self.data_root).rglob('*.tmp.png')))
        self.pyplot.subplots.assert_not_called()

    def test_no_initial_frontiers_still_saves_initial_pair(self):
        self.frontiers = np.empty((0, 2), dtype=int)
        self.run_collection()
        self.assert_pair(1)
        self.assertEqual(self.astar_goals, [])
        self.assertIn('0 frontier steps', self.output.getvalue())
        self.assertNotIn('exception', self.output.getvalue())

    def test_all_candidates_invalid_never_calls_astar_with_none(self):
        self.frontiers = np.array([self.start])
        self.run_collection()
        self.assert_pair(1)
        self.assertEqual(self.astar_goals, [])
        self.assertNotIn('exception', self.output.getvalue())

    def test_last_remaining_valid_candidate_is_used(self):
        self.frontiers = np.array([self.start, [505, 505]])
        self.run_collection()
        self.assert_pair(2)
        np.testing.assert_array_equal(self.astar_goals[-1], [505, 505])
        self.assertIn('1 frontier steps', self.output.getvalue())

    def test_unreachable_candidates_still_save_initial_pair(self):
        self.path_available = False
        self.run_collection()
        self.assert_pair(1)
        self.assertTrue(self.astar_goals)
        self.assertNotIn('exception', self.output.getvalue())

    def test_no_frontiers_after_refresh_stops_cleanly(self):
        self.empty_on_refresh = True
        self.run_collection()
        self.assert_pair(1)
        self.assertEqual(self.astar_goals, [])
        self.assertNotIn('exception', self.output.getvalue())

    def test_exception_after_initial_observation_still_saves_pair(self):
        self.frontier_exception_after_moves = 0
        self.run_collection()
        self.assert_pair(1)
        self.assertIn('simulated frontier detection failure', self.output.getvalue())

    def test_exception_after_moving_saves_latest_observation(self):
        self.frontier_exception_after_moves = 1
        self.run_collection(max_steps=3)
        self.assert_pair(2)
        self.assertIn('simulated frontier detection failure', self.output.getvalue())
        self.assertIn('Saved final training pair sample: 1 frontier steps', self.output.getvalue())

    def test_target_fraction_stop_saves_latest_pair(self):
        self.run_collection(max_steps=3, target_known_fraction=0.02)
        self.assert_pair(2)
        self.assertIn('Reached', self.output.getvalue())

    def test_iou_disabled_with_experiment_artifacts(self):
        self.namespace['collect_opts'].log_iou = True
        self.run_collection()
        self.assert_pair(2)
        self.assertNotIn('exception', self.output.getvalue())

    def test_no_initial_observation_does_not_fabricate_pair(self):
        self.initial_observation_fails = True
        self.run_collection()
        self.assertFalse(self.pair_path('global_gt').exists())
        self.assertFalse(self.pair_path('global_obs').exists())
        self.assertIn('initial observation was not completed', self.output.getvalue())

    def test_failed_final_encode_preserves_initial_pair(self):
        self.frontiers = np.empty((0, 2), dtype=int)
        self.run_collection()
        old_gt = self.pair_path('global_gt').read_bytes()
        old_obs = self.pair_path('global_obs').read_bytes()
        self.mapper.obs_map[504, 504] = 0
        def failed_obs_writer(path, array):
            return False if 'global_obs' in path else write_png(path, array)
        self.namespace['cv2'].imwrite = failed_obs_writer
        with self.assertRaises(IOError):
            self.namespace['save_lama_training_pair'](self.mapper, self.data_root, 'sample', 1)
        self.assertEqual(old_gt, self.pair_path('global_gt').read_bytes())
        self.assertEqual(old_obs, self.pair_path('global_obs').read_bytes())
        self.assertFalse(list(Path(self.data_root).rglob('*.tmp.png')))


class BatchValidationTests(unittest.TestCase):
    def run_validation(self, extra_old=True, missing_current=False, mismatched=False):
        with tempfile.TemporaryDirectory() as temp:
            for folder in ('global_gt', 'global_obs'):
                directory = Path(temp) / 'train_3' / folder
                directory.mkdir(parents=True)
                if extra_old:
                    write_png(directory / 'old.png', np.zeros((4, 4, 3)))
                if not (missing_current and folder == 'global_obs'):
                    shape = (3, 4, 3) if mismatched and folder == 'global_obs' else (4, 4, 3)
                    write_png(directory / 'current.png', np.zeros(shape))
            script = (ROOT / 'collect_lama_pairs.sbatch').read_text(encoding='utf-8')
            invocation = script.split('python -u - "$MAPS_DIR"', 1)[1]
            code = invocation.split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
            collection = {'run_exploration_args': [{'sample_name': 'current'}], 'collection_id': 'current-run'}
            modules = {'cv2': SimpleNamespace(imread=read_png),
                       'matplotlib': SimpleNamespace(use=mock.Mock()),
                       'runpy': SimpleNamespace(run_path=mock.Mock(return_value=collection))}
            output = io.StringIO()
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'argv', ['-', 'maps', temp, '200', '0.25']), redirect_stdout(output):
                exec(compile(code, 'collect_lama_pairs.sbatch', 'exec'), {})
            return output.getvalue()

    def test_older_pairs_do_not_break_current_run_count(self):
        self.assertIn('Validated 1 pairs from collection current-run', self.run_validation())

    def test_old_pairs_do_not_hide_missing_current_observation(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_validation(missing_current=True)
        self.assertIn('current: missing or unreadable', str(raised.exception))

    def test_dimension_mismatch_is_reported(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_validation(mismatched=True)
        self.assertIn('dimensions differ', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
