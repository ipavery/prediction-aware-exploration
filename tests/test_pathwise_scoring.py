"""FrontierPlanner.score_frontiers_pathwise with zero, one and two reachable frontiers.

Only that method is compiled from scripts/sim_utils.py, whose imports need Torch
and the native range library. Its process pool is replaced by an in-process
stand-in, so the method's own code runs unchanged.
"""
import ast
import pathlib
import types
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {'pathwise_index': 1, 'laser_range_m': 20, 'pixel_per_meter': 10}


def load_method():
    tree = ast.parse((ROOT / 'scripts' / 'sim_utils.py').read_text())
    planner = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'FrontierPlanner')
    method = next(node for node in planner.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'score_frontiers_pathwise')
    pools = []

    class Pool(object):
        def __init__(self, processes, initializer, initargs):
            pools.append(self)

        def starmap(self, function, arguments, chunksize):
            return [function(*values) for values in arguments]

        def close(self):
            pass

        def join(self):
            pass

    fields = {'body': [method]}
    if 'type_ignores' in ast.Module._fields:
        fields['type_ignores'] = []
    namespace = dict(np=np, init_worker=lambda: None,
                     mp=types.SimpleNamespace(cpu_count=lambda: 4, Pool=Pool))
    exec(compile(ast.Module(**fields), str(ROOT / 'scripts' / 'sim_utils.py'), 'exec'), namespace)
    return namespace['score_frontiers_pathwise'], pools


@unittest.skipIf(np is None, 'NumPy is needed for the pathwise scoring check')
class PathwiseScoringTests(unittest.TestCase):
    def score(self, frontiers, paths, flooded_sums):
        method, pools = load_method()
        sums = iter(flooded_sums)

        def process_path(path, raycast_range, config, score_mode, mean_map, obs_map, h, w):
            grid = np.zeros((5, 5))
            grid[0, :next(sums)] = 1
            return grid

        planner = types.SimpleNamespace(score_mode='pw-nbv-2d', process_path=process_path)
        result = method(planner, 0, frontiers, paths, None, CONFIG, np.zeros((5, 5)), None, None)
        return result, pools

    def test_no_reachable_frontier_returns_no_goal_without_a_pool(self):
        (centers, costs, most, medium_grid, best, medium, goal), pools = self.score([], [], [])
        self.assertEqual((len(centers), len(costs), best, medium), (0, 0, -1, -1))
        self.assertIsNone(most)
        self.assertIsNone(medium_grid)
        self.assertIsNone(goal)
        self.assertEqual(pools, [])

    def test_one_and_two_frontiers_keep_their_ranking(self):
        path = np.array([[0, 0], [0, 1], [0, 2]])
        result, _ = self.score([[0, 2]], [path.copy()], [2])
        self.assertEqual((result[4], result[5], list(result[6])), (0, -1, [0, 2]))
        result, pools = self.score([[0, 2], [0, 3]], [path.copy(), path.copy()], [2, 4])
        self.assertEqual((result[4], result[5], list(result[6])), (1, 0, [0, 3]))
        self.assertEqual(len(pools), 1)


if __name__ == '__main__':
    unittest.main()
