"""Discover Java inputs and parse them with deterministic repository identities."""
import os
from pathlib import Path

from parser import parse_file

IGNORED_DIRECTORIES = {'.git', 'target', 'build', '.gradle', 'node_modules', '.venv', '__pycache__'}


def discover_java_files(directory):
    files, skipped, diagnostics = [], [], []
    # os.walk reports inaccessible subdirectories through onerror without aborting.
    errors = []
    for folder, directories, names in os.walk(directory, followlinks=False, onerror=errors.append):
        kept = []
        for name in sorted(directories):
            path = Path(folder) / name
            if name in IGNORED_DIRECTORIES or path.is_symlink():
                skipped.append({'file_path': str(path), 'reason': 'ignored directory or symbolic link'})
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix != '.java':
                continue
            if path.is_symlink():
                skipped.append({'file_path': str(path), 'reason': 'symbolic link'})
            else:
                files.append(path.resolve())
    for error in errors:
        diagnostics.append({'kind': 'scan_error', 'file_path': str(error.filename), 'message': str(error)})
    return files, skipped, diagnostics


def discover_inputs(inputs):
    files, roots, skipped, diagnostics = [], [], [], []
    failed = []
    for value in inputs:
        try:
            path = Path(value).expanduser().resolve()
            if path.is_dir():
                roots.append(path)
                discovered, omitted, problems = discover_java_files(path)
                files.extend(discovered)
                skipped.extend(omitted)
                diagnostics.extend(problems)
            else:
                files.append(path)
        except (OSError, ValueError, RuntimeError) as error:
            failed.append({'file_path': str(value), 'has_error': True, 'classes': [],
                           'package': '', 'imports': [],
                           'diagnostics': [{'kind': 'file_error', 'message': str(error)}]})
    # Directory mode uses a common root, preserving identities across repeated runs.
    root = None
    if roots:
        anchors = roots + [path.parent for path in files if not any(path.is_relative_to(r) for r in roots)]
        root = Path(os.path.commonpath(anchors))
    for records in (skipped, diagnostics):
        for record in records:
            if root and record.get('file_path'):
                try:
                    record['file_path'] = Path(record['file_path']).relative_to(root).as_posix()
                except ValueError:
                    pass
    return {
        'failed': failed,
        'source_paths': [str(path) for path in sorted(set(files))],
        'root': str(root) if root else None,
        'discovered': len(set(files)),
        'skipped': sorted(skipped, key=str),
        'diagnostics': sorted(diagnostics, key=str),
    }


def index_inputs(inputs):
    discovery = discover_inputs(inputs)
    results = [parse_file(path, relative_to=discovery['root']) for path in discovery['source_paths']]
    results.extend(discovery.pop('failed'))
    return dict(discovery, files=results)
