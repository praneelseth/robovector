"""Prepare an attributed, instruction-only AdvBench/Alpaca calibration adaptation.

Never downloads or reads the withheld mixed harmful_train pool. Network access is
explicit and restricted to the llm-attacks commit API, LICENSE and AdvBench CSV.
No model use occurs here; val/test instructions are read solely for deduplication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import unicodedata
import urllib.request
from pathlib import Path

NORMALIZATION = 'Unicode NFKC, casefold, whitespace collapse; exact normalized equality only'
DEFAULT_OUT = Path('data/chat_refusal/calibration_advbench_v1')
DEFAULT_POOLS = Path('data/chat_refusal/downloads')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize_instruction(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Instruction must be a nonempty string')
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def instruction_hash(text):
    return digest(normalize_instruction(text).encode('utf-8'))


def download_source(output):
    raw_dir = Path(output) / 'raw'
    if raw_dir.exists():
        raise FileExistsError('Raw directory already exists; preserve it and prepare offline')
    api = 'https://api.github.com/repos/llm-attacks/llm-attacks/commits/main'
    request = urllib.request.Request(api, headers={'User-Agent': 'RoboVector-calibration-provenance'})
    commit = json.load(urllib.request.urlopen(request, timeout=45))['sha']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Unexpected source commit')
    files = {}
    for name, relative in [('LICENSE', 'LICENSE'), ('harmful_behaviors.csv', 'data/advbench/harmful_behaviors.csv')]:
        url = f'https://raw.githubusercontent.com/llm-attacks/llm-attacks/{commit}/{relative}'
        files[name] = (urllib.request.urlopen(url, timeout=45).read(), url)
    raw_dir.mkdir(parents=True)
    metadata = dict(commit_api_url=api, source_commit=commit, files={})
    for name, (raw, url) in files.items():
        (raw_dir / name).write_bytes(raw)
        metadata['files'][name] = dict(url=url, sha256=digest(raw))
    (raw_dir / 'download_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def eligible_rows(rows, field, excluded_hashes):
    seen, eligible = set(), []
    counts = dict(raw=len(rows), internal_duplicates=0, excluded_overlap=0)
    for index, row in enumerate(rows):
        text = row[field]
        key = instruction_hash(text)
        if key in seen:
            counts['internal_duplicates'] += 1
            continue
        seen.add(key)
        if key in excluded_hashes:
            counts['excluded_overlap'] += 1
            continue
        eligible.append(dict(source_index=index, instruction=text, instruction_sha256=key))
    counts['eligible'] = len(eligible)
    return eligible, counts


def sample_rows(rows, train_count=128, selection_count=32, seed=42):
    if train_count < 1 or selection_count < 1 or len(rows) < train_count + selection_count:
        raise ValueError('Insufficient eligible examples or invalid requested split size')
    ordered = sorted(rows, key=lambda r: r['source_index'])
    random.Random(seed).shuffle(ordered)
    return ordered[:train_count], ordered[train_count:train_count + selection_count]


def prepare(output=DEFAULT_OUT, pools=DEFAULT_POOLS, prior_calibration_files=(), *,
            train_count=128, selection_count=32, screen_count=8, seed=42):
    output, pools = Path(output), Path(pools)
    if any((output / name).exists() for name in ('manifest.json', 'train.jsonl', 'selection.jsonl', 'README.md')):
        raise FileExistsError('Calibration outputs already exist; do not overwrite frozen samples')
    if not 0 < screen_count < selection_count:
        raise ValueError('Candidate screen must leave a separate confirmation subset')
    source = json.loads((output / 'raw/download_provenance.json').read_text(encoding='utf-8'))
    for name, meta in source['files'].items():
        if digest((output / 'raw' / name).read_bytes()) != meta['sha256']:
            raise ValueError('Raw source hash mismatch')
    license_text = (output / 'raw/LICENSE').read_text(encoding='utf-8')
    if 'MIT License' not in license_text or 'Copyright (c) 2023 Andy Zou' not in license_text:
        raise ValueError('Expected source MIT attribution missing')
    exclusion_files = [pools / f'{label}_{split}.json'
                       for label in ('harmful', 'harmless') for split in ('val', 'test')]
    exclusion_files += sorted({Path(p) for p in prior_calibration_files})
    excluded, exclusion_provenance = set(), []
    for path in exclusion_files:
        if path.name == 'harmful_train.json':
            raise PermissionError('Mixed harmful training data must remain withheld')
        raw = path.read_bytes()
        rows = ([json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
                if path.suffix == '.jsonl' else json.loads(raw))
        keys = {instruction_hash(row['instruction']) for row in rows}
        excluded.update(keys)
        exclusion_provenance.append(dict(path=path.as_posix(), sha256=digest(raw), n_rows=len(rows),
                                         purpose='Instruction hash exclusion only; no model evaluation'))
    csv_path = output / 'raw/harmful_behaviors.csv'
    with csv_path.open(encoding='utf-8', newline='') as handle:
        harmful_source = list(csv.DictReader(handle))
    # Only goal is propagated; the CSV target column is never selected or generated.
    harmful, harmful_counts = eligible_rows(harmful_source, 'goal', excluded)
    harmless_path = pools / 'harmless_train.json'
    harmless_raw = harmless_path.read_bytes()
    harmless_provenance = json.loads((pools / 'harmless_train.provenance.json').read_text(encoding='utf-8'))
    if digest(harmless_raw) != harmless_provenance['sha256']:
        raise ValueError('Harmless source hash mismatch')
    harmless, harmless_counts = eligible_rows(json.loads(harmless_raw), 'instruction',
                                               excluded | {r['instruction_sha256'] for r in harmful})
    training, selection = [], []
    for label, pool in [('harmful', harmful), ('harmless', harmless)]:
        train, select = sample_rows(pool, train_count, selection_count, seed)
        for split_name, chosen, destination in [('train', train, training), ('selection', select, selection)]:
            for position, record in enumerate(chosen):
                prefix = 'advbench' if label == 'harmful' else 'alpaca_harmless_train'
                destination.append(dict(record, id=f'{prefix}_{record["source_index"]:05d}',
                    source_record_id=f'{prefix}:{record["source_index"]}', harm_label=label,
                    split=split_name, selection_role=(None if split_name == 'train' else
                        'candidate_screen' if position < screen_count else 'confirmation'),
                    source_url=(source['files']['harmful_behaviors.csv']['url'] if label == 'harmful'
                                else harmless_provenance['url']),
                    license='MIT' if label == 'harmful' else 'CC-BY-NC-4.0; split packaging Apache-2.0'))
    all_hashes = [r['instruction_sha256'] for r in training + selection]
    if len(set(all_hashes)) != len(all_hashes) or set(all_hashes) & excluded:
        raise ValueError('Selected instruction contamination or duplicate')
    manifest = dict(version='licensed-calibration-advbench-v1', seed=seed, normalization=NORMALIZATION,
        adaptation='Source-cleared AdvBench goals + original harmless_train; NOT exact Arditi split',
        sampling='Sort eligible source indices, shuffle separately per label with random.Random(seed), train then selection',
        source=source, harmless_source=dict(path=harmless_path.as_posix(), **harmless_provenance),
        licensing=dict(harmful='MIT, Copyright (c) 2023 Andy Zou; retain raw/LICENSE',
            harmless='Stanford Alpaca CC-BY-NC-4.0, research/noncommercial; preserve attribution and restrictions',
            harmless_license_url=harmless_provenance['licenses']['Alpaca_CC-BY-NC-4.0.txt'],
            alpaca_source='https://github.com/tatsu-lab/stanford_alpaca',
            arditi_source='https://github.com/andyrdt/refusal_direction'),
        withheld='Mixed harmful_train is neither downloaded nor read',
        counts=dict(harmful=harmful_counts, harmless=harmless_counts, exclusion_unique_hashes=len(excluded)),
        exclusion_files=exclusion_provenance,
        selection_roles=dict(candidate_screen_per_label=screen_count,
                             confirmation_per_label=selection_count-screen_count), splits={})
    for name, rows in [('train', training), ('selection', selection)]:
        raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
        (output / f'{name}.jsonl').write_bytes(raw)
        manifest['splits'][name] = dict(count=len(rows), sha256=digest(raw), record_ids=[r['id'] for r in rows],
                                       instruction_sha256=[r['instruction_sha256'] for r in rows])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Licensed calibration adaptation v1\n\n'
        'This is a source-cleared adaptation, not the exact Arditi paper split. '
        'Harmful instructions are AdvBench goals only from llm-attacks, MIT licensed '
        '(Copyright (c) 2023 Andy Zou); retain raw/LICENSE. No CSV target text is '
        'included in selected records. Harmless instructions come only from the original '
        'Arditi harmless_train pool, derived from Stanford Alpaca under CC-BY-NC-4.0 '
        '(noncommercial research; preserve attribution), with Apache-2.0 split packaging.\n\n'
        'train.jsonl: 128 harmful + 128 harmless. selection.jsonl: 32 + 32, ordered '
        'per label; first 8 per label candidate_screen, remaining 24 per label '
        'confirmation. Do not use confirmation outputs for candidate screening. '
        'These role names reserve data; preparation itself makes no model calls.\n\n'
        'Manifest records source URLs, pinned commit, raw and split hashes, record IDs, '
        'licenses, deterministic sampling, and overlap exclusions. All existing val/test '
        'instruction pools and enumerated prior calibration files were read only for '
        'normalized exact-instruction exclusion; test prompts were not evaluated. '
        'This does not remove paraphrases or semantic overlaps. Mixed harmful_train '
        'was neither downloaded nor read.\n\n'
        'Sources: https://github.com/llm-attacks/llm-attacks ; '
        'https://github.com/tatsu-lab/stanford_alpaca ; '
        'https://github.com/andyrdt/refusal_direction .\n\n'
        'Reproduce offline in a fresh output directory containing an identical raw/ '
        'copy: python -m data.chat_refusal.prepare_calibration --output <directory> '
        '--prior-calibration <each manifest-listed prior calibration JSONL>. '
        'Downloads require explicit --download. No implicit network or model download.\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--pools', type=Path, default=DEFAULT_POOLS)
    parser.add_argument('--prior-calibration', type=Path, action='append', default=[])
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download_source(args.output)
    manifest = prepare(args.output, args.pools, args.prior_calibration)
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
