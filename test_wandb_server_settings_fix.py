"""No W&B, Torch, GPU, or network needed for these regression checks."""
from pathlib import Path
import tempfile
import unittest

from fix_wandb_server_settings import OLD, NEW, patched_source, patch_file


class NullIntrospectionTests(unittest.TestCase):
    def execute(self, expression, response):
        source = 'def probe(res):\n    return [field.get("name", "") for field in ' + expression + ']\n'
        namespace = {}
        exec(source, namespace)
        return namespace['probe'](response)

    def test_reproduces_observed_crash(self):
        with self.assertRaises(AttributeError):
            self.execute(OLD, {'ServerSettingsType': None})

    def test_fixed_null_and_missing_type(self):
        self.assertEqual(self.execute(NEW, {'ServerSettingsType': None}), [])
        self.assertEqual(self.execute(NEW, {}), [])

    def test_existing_fields_preserved(self):
        response = {'ServerSettingsType': {'fields': [{'name': 'exampleSetting'}]}}
        self.assertEqual(self.execute(NEW, response), self.execute(OLD, response))

    def test_backup_and_idempotency(self):
        source = 'def probe(res):\n    return ' + OLD + '\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'internal_api.py'
            path.write_text(source)
            changed, backup = patch_file(path)
            self.assertTrue(changed)
            self.assertEqual(backup.read_text(), source)
            self.assertIn(NEW, path.read_text())
            self.assertFalse(patch_file(path)[0])
            self.assertEqual(backup.read_text(), source)

    def test_unexpected_source_fails_without_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'internal_api.py'
            path.write_text('x = 1\n')
            with self.assertRaises(ValueError):
                patch_file(path)
            self.assertEqual(path.read_text(), 'x = 1\n')
            self.assertFalse(path.with_name(path.name + '.before-png-nullfix').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
