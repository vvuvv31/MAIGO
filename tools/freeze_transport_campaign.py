#!/usr/bin/env python3
"""Freeze the actual working tree (including untracked build dependencies)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    manifest = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'files': {}, 'inputs': {}}
    (out / 'worktree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=root))
    (out / 'status.txt').write_bytes(subprocess.check_output(['git', 'status', '--porcelain=v1', '-uall'], cwd=root))
    # Explicit build dependency roots include untracked .hpp/.inc/test/tool files.
    for name in ('CMakeLists.txt', 'AGENTS.md', 'src', 'include', 'tests', 'tools', 'cmake', 'config', 'data'):
        src = root / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, source / name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(src, source / name)
    for f in sorted(source.rglob('*')):
        if f.is_file():
            manifest['files'][str(f.relative_to(source))] = {'sha256': sha(f), 'bytes': f.stat().st_size}
    text = a.config.read_text()
    (out / 'original.yaml').write_text(text)
    frozen = []
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            path = Path(value.strip().strip('\"\''))
            if path.is_absolute() and path.is_file():
                try:
                    rel = path.relative_to(root / 'data')
                    target = source / 'data' / rel
                except ValueError:
                    target = out / 'inputs' / key / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                manifest['inputs'][key] = {'original': str(path), 'frozen': str(target), 'sha256': sha(target)}
                line = f'{key}: {target}'
        frozen.append(line)
    (out / 'config.yaml').write_text('\n'.join(frozen) + '\n')
    manifest['config_sha256'] = sha(out / 'config.yaml')
    (out / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
