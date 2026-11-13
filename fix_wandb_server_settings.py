#!/usr/bin/env python3
"""Back up and fix one legacy W&B null-introspection expression.

Run once from the activated pipe Conda environment on the login node.
Does not make network requests, install packages, or change training code.
"""
import ast
import importlib.util
from pathlib import Path
import shutil
import sys


OLD = 'res.get("ServerSettingsType", {}).get("fields", [{}])'
NEW = '(res.get("ServerSettingsType") or {}).get("fields", [])'


def patched_source(source):
    if OLD not in source:
        if NEW in source:
            return source, False
        raise ValueError('Expected legacy expression not found. No change made; this SDK needs separate inspection.')
    if source.count(OLD) != 1:
        raise ValueError('More than one matching expression. No change made.')
    result = source.replace(OLD, NEW, 1)
    ast.parse(result)
    return result, True


def patch_file(path):
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    result, changed = patched_source(source)
    backup = path.with_name(path.name + '.before-png-nullfix')
    if not changed:
        return False, backup
    if backup.exists():
        if backup.read_text(encoding='utf-8') != source:
            raise ValueError('A different backup already exists. No files changed; inspect it before proceeding.')
    else:
        shutil.copy2(str(path), str(backup))
    path.write_text(result, encoding='utf-8')
    return True, backup


def main():
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python fix_wandb_server_settings.py (inside your activated pipe environment)')
    spec = importlib.util.find_spec('wandb')
    if spec is None or spec.origin is None:
        raise SystemExit('W&B is not installed in this Python environment. Activate pipe first.')
    target = Path(spec.origin).parent / 'sdk' / 'internal' / 'internal_api.py'
    if not target.is_file():
        raise SystemExit('Legacy SDK source file not found. No change made.')
    changed, backup = patch_file(target)
    print('Python environment:', sys.executable)
    print('W&B source:', target)
    print('Applied null-introspection fix.' if changed else 'Fix already present; no changes needed.')
    print('Original backup:', backup)
    print('Submit a NEW smoke job so its W&B processes load the corrected source.')


if __name__ == '__main__':
    main()
