import pytest
import json
from pathlib import Path

from data.chat_refusal.prepare_calibration import digest, eligible_rows, instruction_hash, prepare, sample_rows


def test_unicode_case_and_whitespace_dedup_exclusion():
    rows = [{'goal': '  HELLO\tWorld '}, {'goal': 'hello world'}, {'goal': 'Fullwidth Ａ'}, {'goal': 'other'}]
    eligible, counts = eligible_rows(rows, 'goal', {instruction_hash('fullwidth a')})
    assert [r['source_index'] for r in eligible] == [0, 3]
    assert counts == dict(raw=4, internal_duplicates=1, excluded_overlap=1, eligible=2)
    assert all('target' not in r and set(r) == {'source_index','instruction','instruction_sha256'} for r in eligible)


def test_sampling_is_reproducible_disjoint_and_source_order_independent():
    rows = [dict(source_index=i, instruction=str(i)) for i in range(20)]
    train, selection = sample_rows(rows, 8, 4, 42)
    assert (train, selection) == sample_rows(list(reversed(rows)), 8, 4, 42)
    assert {r['source_index'] for r in train}.isdisjoint({r['source_index'] for r in selection})
    assert len(train) == 8 and len(selection) == 4
    with pytest.raises(ValueError):
        sample_rows(rows, 16, 8)


def test_end_to_end_does_not_read_withheld_pool_or_propagate_target(tmp_path):
    output, pools = tmp_path / 'calibration', tmp_path / 'pools'
    raw = output / 'raw'
    raw.mkdir(parents=True)
    pools.mkdir()
    license_bytes = b'MIT License\nCopyright (c) 2023 Andy Zou\n'
    csv_bytes = ('goal,target\n' + ''.join(f'goal {i},TARGET_MUST_NOT_PROPAGATE\n' for i in range(10))).encode()
    files = {'LICENSE': license_bytes, 'harmful_behaviors.csv': csv_bytes}
    for name, content in files.items():
        (raw/name).write_bytes(content)
    (raw/'download_provenance.json').write_text(json.dumps(dict(source_commit='a'*40,
        files={name: dict(url='https://example.invalid/'+name, sha256=digest(content)) for name,content in files.items()})))
    excluded_paths = []
    for label in ('harmful','harmless'):
        for split in ('val','test'):
            path = pools/f'{label}_{split}.json'
            path.write_text(json.dumps([dict(instruction='goal 0')]))
            excluded_paths.append(path)
    prior = tmp_path/'calibration.jsonl'
    prior.write_text(json.dumps(dict(instruction='goal 1'))+'\n')
    harmless_raw = json.dumps([dict(instruction=f'benign {i}') for i in range(10)]).encode()
    (pools/'harmless_train.json').write_bytes(harmless_raw)
    (pools/'harmless_train.provenance.json').write_text(json.dumps(dict(sha256=digest(harmless_raw),
        url='https://example.invalid/harmless', licenses={'Alpaca_CC-BY-NC-4.0.txt':'https://example.invalid/license'})))
    before = {path:path.read_bytes() for path in excluded_paths}
    # No harmful_train exists: successful preparation cannot depend on reading it.
    manifest = prepare(output,pools,[prior],train_count=2,selection_count=2,screen_count=1)
    records = [json.loads(line) for name in ('train','selection')
               for line in (output/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == 8
    assert len({r['instruction_sha256'] for r in records}) == 8
    assert not {'goal 0','goal 1'} & {r['instruction'] for r in records}
    assert 'TARGET_MUST_NOT_PROPAGATE' not in (output/'train.jsonl').read_text()
    assert 'TARGET_MUST_NOT_PROPAGATE' not in (output/'selection.jsonl').read_text()
    selected = [r for r in records if r['split']=='selection']
    assert sum(r['selection_role']=='candidate_screen' for r in selected) == 2
    assert sum(r['selection_role']=='confirmation' for r in selected) == 2
    assert all(path.read_bytes()==content for path,content in before.items())
    assert manifest['counts']['harmful']['excluded_overlap'] == 2
    with pytest.raises(FileExistsError):
        prepare(output,pools,[prior],train_count=2,selection_count=2,screen_count=1)
