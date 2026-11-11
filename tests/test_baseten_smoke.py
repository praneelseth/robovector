import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def server_module():
    spec = importlib.util.spec_from_file_location('smoke_server', ROOT / 'deploy/qwen_smoke/model/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_replay_does_not_generate_twice(tmp_path):
    model = server_module().Model(data_dir=tmp_path)
    calls = []
    model._run = lambda: calls.append(1) or {'status': 'fixture'}
    first = model.predict({'operation': 'smoke-v1'})
    assert model.predict({'operation': 'smoke-v1'}) is first
    assert calls == [1]


@pytest.mark.parametrize('payload', [{}, {'operation': 'anything'}, {'operation': 'smoke-v1', 'max_tokens': 999999}])
def test_smoke_rejects_unbounded_requests(tmp_path, payload):
    model = server_module().Model(data_dir=tmp_path)
    with pytest.raises(ValueError, match='Only the fixed'):
        model.predict(payload)
