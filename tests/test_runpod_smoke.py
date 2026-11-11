import hashlib
import json
import zipfile
from types import SimpleNamespace

import pytest

from deploy.runpod.smoke import require_remote_gpu
from scripts.package_runpod_smoke import package


def test_bundle_is_allowlisted_and_matches_manifest(tmp_path):
    destination = tmp_path / 'bundle.zip'
    manifest = package(destination)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == set(manifest) | {'bundle-manifest.json'}
        for name, sha in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
            assert not name.endswith(('.env', '.safetensors', '.pt'))
            assert name.startswith(('src/', 'data/', 'deploy/'))
        tasks = [json.loads(s) for s in archive.read('data/tasks.jsonl').splitlines()]
        assert {t['task_id'] for t in tasks} == {'burner', 'burner_safe'}
        assert tasks[0]['image_ids'] == tasks[1]['image_ids']
        assert 'different clear spot' in tasks[1]['instruction']
    with pytest.raises(FileExistsError):
        package(destination)


def test_laptop_rejected_before_cuda_or_download(monkeypatch):
    monkeypatch.setattr('deploy.runpod.smoke.platform.system', lambda: 'Windows')
    with pytest.raises(RuntimeError, match='laptop'):
        require_remote_gpu(SimpleNamespace())


def test_remote_requires_sufficient_vram(monkeypatch):
    monkeypatch.setattr('deploy.runpod.smoke.platform.system', lambda: 'Linux')
    monkeypatch.setenv('RUNPOD_POD_ID', 'fixture')
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 1024**3))
    with pytest.raises(RuntimeError, match='48 GB'):
        require_remote_gpu(SimpleNamespace(cuda=cuda))
