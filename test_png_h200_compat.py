"""Check runner API wiring without importing GPU libraries or submitting jobs.

The fake trainer exercises configuration/resume/completion behavior only;
the real H200 smoke jobs remain the optimizer/kernel integration test.
"""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest

RUNNER = Path(__file__).parent / 'lama/bin/train_png_maps.py'


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def extract(name, class_name=None):
    tree = ast.parse(RUNNER.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), 'exec')


@contextmanager
def temporary_run():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        try:
            yield
        finally:
            os.chdir(previous)


class RunnerCompatibilityTests(unittest.TestCase):
    def run_main(self, resume=None, base=None, smoke=True, incomplete=False, version='1.9.5'):
        model = SimpleNamespace(train_set=[1], val_set=[1], png_optimizer_steps=[0, 0])
        callback = SimpleNamespace(best_model_path='models/best.ckpt', last_model_path='models/last.ckpt')
        calls = {}

        class Trainer:
            def __init__(self, **kwargs):
                calls['trainer'] = kwargs

            def fit(self, supplied_model, ckpt_path=None):
                calls['fit_model'] = supplied_model
                calls['resume'] = ckpt_path
                model.png_optimizer_steps[:] = [2, 0 if incomplete else 2]
                for path in (callback.best_model_path, callback.last_model_path,
                             'png_samples/epoch_000/val_000.png', 'validation.jsonl'):
                    target = Path(path)
                    target.parent.mkdir(exist_ok=True, parents=True)
                    target.touch()

        noop = lambda *args, **kwargs: None
        options = dict(accelerator='gpu', devices=1, precision=32, max_epochs=1,
                       resume_from_checkpoint=resume)
        config = Config(pretrained_generator_path=base, seed=42,
                        visualizer=Config(outdir='previews'), location=Config(tb_dir='tb'),
                        trainer=Config(kwargs=Config(options)))
        namespace = dict(os=os, sys=sys, json=json, platform=platform, Path=Path,
                         LOG=SimpleNamespace(info=noop),
                         np=SimpleNamespace(__version__='fixture'),
                         torch=SimpleNamespace(__version__='2.0.1+cu118',
                             version=SimpleNamespace(cuda='11.8'),
                             backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 8700))),
                         pl=SimpleNamespace(__version__=version, seed_everything=noop, Trainer=Trainer),
                         OmegaConf=SimpleNamespace(to_container=lambda value, **kwargs: dict(value), save=noop),
                         CSVLogger=noop, TensorBoardLogger=noop,
                         WandbLogger=lambda **kwargs: self.fail('disabled W&B must not create a remote logger'),
                         ModelCheckpoint=lambda **kwargs: callback, PngTrainingModule=lambda config: model,
                         initialize_from_checkpoint=lambda *args, **kwargs: dict(
                             discriminator_initialized=True, sha256='fixture'),
                         wandb=SimpleNamespace(__version__='fixture', init=noop, finish=noop))
        exec(extract('main'), namespace)
        previous = {key: os.environ.get(key) for key in ('WANDB_MODE', 'SMOKE')}
        os.environ.update(WANDB_MODE='disabled', SMOKE='1' if smoke else '0')
        try:
            namespace['main'](config)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return calls

    def test_resume_goes_to_fit_and_not_trainer(self):
        with temporary_run():
            calls = self.run_main(resume='/fixture/last.ckpt')
            self.assertNotIn('resume_from_checkpoint', calls['trainer'])
            self.assertEqual(calls['resume'], '/fixture/last.ckpt')
            self.assertEqual(calls['trainer']['devices'], 1)
            self.assertEqual(calls['trainer']['precision'], 32)
            self.assertEqual(json.loads(Path('initialization.json').read_text())['kind'], 'resume')

    def test_rejects_base_and_resume_before_training(self):
        with temporary_run(), self.assertRaisesRegex(ValueError, 'base initialization'):
            self.run_main(base='/fixture/base.ckpt', resume='/fixture/last.ckpt')

    def test_rejects_wrong_lightning_version(self):
        with temporary_run(), self.assertRaisesRegex(RuntimeError, 'Lightning 1.9'):
            self.run_main(version='2.0.0')

    def test_smoke_records_both_optimizer_updates_and_artifacts(self):
        with temporary_run():
            self.run_main()
            summary = json.loads(Path('training_complete.json').read_text())
            self.assertEqual((summary['generator_steps'], summary['discriminator_steps']), (2, 2))
            self.assertTrue(summary['validation_recorded'])
            self.assertEqual(summary['preview_count'], 1)

    def test_missing_discriminator_updates_cannot_report_success(self):
        with temporary_run():
            with self.assertRaisesRegex(RuntimeError, 'Smoke run did not complete'):
                self.run_main(incomplete=True)
            self.assertFalse(Path('training_complete.json').exists())

    def test_rank_uses_world_size(self):
        namespace = {}
        exec(extract('get_ddp_rank', 'PngTrainingModule'), namespace)
        for size, rank, expected in [(1, 0, None), (2, 1, 1)]:
            obj = SimpleNamespace(trainer=SimpleNamespace(world_size=size, global_rank=rank))
            self.assertEqual(namespace['get_ddp_rank'](obj), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
