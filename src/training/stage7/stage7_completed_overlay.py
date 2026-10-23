"""Authenticate Notebook 01's two-member successor without changing its base cache."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile

PREPARATION = 'Poker Assistant Model/Stage 6 - Qualified Layers Improvement/src/stage6_empirical_preparation_v2.py'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def identity(path):
    with Path(path).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': digest}

def raw_identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def safe(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
            'Unsafe overlay path')
    return path

def land(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, 'Different existing overlay member')
    else:
        with path.open('xb') as target:
            target.write(raw)

def activate(base_workspace, destination, base_manifest, overlay_path, expected_overlay):
    """Called only after the base notebook's authenticated extraction cell.

    Unchanged members share read-only input bytes through hardlinks; no source
    or model is rewritten. The prepared/scores directories use the new plan ID.
    """
    base_workspace, destination = Path(base_workspace), Path(destination)
    require(base_workspace.resolve() != destination.resolve(), 'Overlay needs a separate workspace')
    require(identity(overlay_path) == expected_overlay, 'Overlay ZIP identity differs')
    with zipfile.ZipFile(overlay_path) as bundle:
        names = bundle.namelist()
        overlay = json.loads(bundle.read('overlay_manifest.json'))
        require(overlay['schema_version'] == 'stage7_completed_code_overlay_v1', 'Wrong overlay schema')
        require(len(names) == len(set(names)) and set(names) == set(overlay['members']) | {'overlay_manifest.json'},
                'Overlay inventory differs')
        payload = {}
        for name, expected in overlay['members'].items():
            safe(name)
            info = bundle.getinfo(name)
            require((info.external_attr >> 16) & 0o170000 != 0o120000, 'Overlay symlink rejected')
            raw = bundle.read(name)
            require(raw_identity(raw) == expected, 'Overlay member identity differs')
            payload[name] = raw
    require(raw_identity(canonical(base_manifest)) == overlay['base_manifest_canonical'], 'Wrong base manifest')
    plan_name = base_manifest['plan_relative_path']
    require(set(overlay['replacements']) == {PREPARATION, plan_name}, 'Unexpected replacement scope')
    require(set(payload) == {PREPARATION, plan_name, 'stage7_completed_overlay.py'}, 'Unexpected payload scope')
    for name, change in overlay['replacements'].items():
        require(change['before'] == base_manifest['members'][name], 'Replacement predecessor differs')
        require(identity(base_workspace / name) == change['before'], 'Base predecessor bytes differ')
        require(raw_identity(payload[name]) == change['after'], 'Replacement successor differs')
    old_plan = json.loads((base_workspace / plan_name).read_bytes())
    new_plan = json.loads(payload[plan_name])
    expected_plan = copy.deepcopy(old_plan)
    expected_plan['code_files'][PREPARATION] = overlay['replacements'][PREPARATION]['after']
    require(canonical(new_plan) == canonical(expected_plan), 'Plan change exceeds repaired code identity')
    require(new_plan['code_files'][PREPARATION] == raw_identity(payload[PREPARATION]), 'Plan/source mismatch')
    require(raw_identity(payload[plan_name])['sha256'] == overlay['plan_sha256'], 'Wrong plan SHA')
    manifest = copy.deepcopy(base_manifest)
    manifest['plan_sha256'] = overlay['plan_sha256']
    for name, change in overlay['replacements'].items():
        manifest['members'][name] = change['after']
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['members'].items():
        safe(name)
        target = destination / name
        if name in overlay['replacements']:
            land(target, payload[name])
            continue
        source = base_workspace / name
        require(source.is_file() and not source.is_symlink() and source.stat().st_size == expected['bytes'],
                'Authenticated base member is missing or changed size')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                # Cross-device filesystems need a real copy, verified once.
                with source.open('rb') as inp, target.open('xb') as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
                require(identity(target) == expected, 'Copied cache member differs')
        elif not os.path.samefile(source, target):
            require(identity(target) == expected, 'Existing cache member differs')
    land(destination / 'stage7_overlay_activation_receipt.json', canonical({
        'schema_version': 'stage7_completed_overlay_activation_v1', 'overlay': expected_overlay,
        'base_workspace': str(base_workspace), 'plan_sha256': overlay['plan_sha256'],
        'replacements': overlay['replacements'], 'unchanged_members': len(manifest['members']) - 2}))
    return destination, manifest
