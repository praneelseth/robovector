import hashlib
import json
from pathlib import Path
import urllib.request

import pytest

from deploy.runpod.tonight import load_calibration, main, verify_freeze
from scripts.prepare_calibration import prepare


def write_input(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'calibration.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    metadata = {'output': {'file': path.name, 'count': len(rows),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    path.with_name('provenance.json').write_text(json.dumps(metadata), encoding='utf-8')
    return path


def valid_rows():
    return [dict(id=label, instruction='Example instruction', harm_label=label, source_split='val')
            for label in ('harmful', 'harmless')]


@pytest.mark.parametrize('identifier', ['../escape', '/absolute', 'a/b', 'a\\b', '', '.', 'a b'])
def test_unsafe_ids_rejected(tmp_path, identifier):
    rows = valid_rows()
    rows[0]['id'] = identifier
    with pytest.raises(ValueError, match='ID'):
        load_calibration(write_input(tmp_path, rows))


def test_duplicates_and_unbalanced_rejected(tmp_path):
    rows = valid_rows()
    rows[1]['id'] = rows[0]['id']
    with pytest.raises(ValueError, match='ID'):
        load_calibration(write_input(tmp_path, rows))
    with pytest.raises(ValueError, match='balanced'):
        load_calibration(write_input(tmp_path, valid_rows()[:1]))


@pytest.mark.parametrize('field', ['instruction', 'harm_label', 'source_split'])
def test_missing_record_metadata(tmp_path, field):
    rows = valid_rows()
    del rows[0][field]
    with pytest.raises(ValueError):
        load_calibration(write_input(tmp_path, rows))


def test_provenance_required_and_hash_checked(tmp_path):
    path = write_input(tmp_path, valid_rows())
    path.write_text(path.read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        load_calibration(path)
    path.with_name('provenance.json').unlink()
    with pytest.raises(FileNotFoundError):
        load_calibration(path)


def test_frozen_inputs_and_hash_failure(tmp_path):
    verify_freeze()
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'input.txt').write_bytes(b'hello\r\n')
    digest = hashlib.sha256(b'hello\n').hexdigest()
    (tmp_path / 'configs/pilot_burner_v2.freeze.json').write_text(
        json.dumps({'sha256': {'input.txt': digest}}))
    verify_freeze(tmp_path)
    (tmp_path / 'input.txt').write_text('changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        verify_freeze(tmp_path)


def test_invalid_input_precedes_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr('deploy.runpod.tonight.verify_freeze', lambda: {})
    with pytest.raises(FileNotFoundError):
        main(['--output', str(tmp_path / 'run'), '--model-dir', 'missing-model',
              '--calibration-input', str(tmp_path / 'missing.jsonl')])
    assert not (tmp_path / 'run').exists()


def test_prepare_deterministic_local_validation_only(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Preparation must never access the network')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    pools = tmp_path / 'pools'
    pools.mkdir()
    manifest = {}
    for label in ('harmful', 'harmless'):
        name = f'{label}_val'
        path = pools / f'{name}.json'
        path.write_text(json.dumps([{'instruction': f'{label} example {i}'} for i in range(40)]))
        manifest[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=40,
                              licenses={'LICENSE.txt': 'fixture'}, revision='fixture')
    (pools / 'LICENSE.txt').write_text('Test fixture license')
    sentinel = pools / 'harmful_train.json'
    sentinel.write_bytes(b'untouched')
    (pools / 'harmless_test.json').write_bytes(b'untouched test')
    before = {p.name: p.read_bytes() for p in pools.iterdir()}
    manifest_path = tmp_path / 'sources.json'
    manifest_path.write_text(json.dumps(manifest))
    first = prepare(pools, tmp_path / 'first', source_manifest=manifest_path)
    second = prepare(pools, tmp_path / 'second', source_manifest=manifest_path)
    assert first.read_bytes() == second.read_bytes()
    rows, _ = load_calibration(first)
    assert len(rows) == 64
    assert {p.name: p.read_bytes() for p in pools.iterdir()} == before
    with pytest.raises(FileExistsError):
        prepare(pools, tmp_path / 'first', source_manifest=manifest_path)
    (pools / 'harmful_val.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source hash'):
        prepare(pools, tmp_path / 'third', source_manifest=manifest_path)
