"""Prepare a deterministic exploratory calibration set from local validation pools only."""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(directory, output, *, source_manifest=None):
    directory, output = Path(directory), Path(output)
    manifest = json.loads(Path(source_manifest or ROOT / 'data/chat_refusal/SOURCES.json').read_text(encoding='utf-8'))
    records, sources = [], {}
    for label in ('harmful', 'harmless'):
        name = f'{label}_val'
        path = directory / f'{name}.json'
        metadata = manifest[name]
        if sha256(path) != metadata['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        for notice in metadata['licenses']:
            if not (directory / notice).is_file():
                raise ValueError(f'Missing local license notice: {notice}')
        pool = json.loads(path.read_text(encoding='utf-8'))
        if len(pool) != metadata['count'] or len(pool) < 32:
            raise ValueError(f'Invalid validation pool: {name}')
        for index in random.Random(42).sample(range(len(pool)), 32):
            instruction = pool[index].get('instruction')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('Empty validation instruction')
            records.append(dict(id=f'{name}_{index:05d}', instruction=instruction,
                                harm_label=label, source_split='val', source_index=index))
        sources[name] = dict(metadata, path=str(path.resolve()))
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'calibration.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8', newline='\n')
    provenance = dict(seed=42, per_label=32, purpose='exploratory_validation_calibration',
                      sources=sources, output=dict(file=target.name, sha256=sha256(target), count=len(records)))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        'Exploratory validation calibration: 32 instructions per label, seed 42.\n'
        'Only local validation pools were read; no network access, training or test data modification.\n'
        'This is a new deterministic sample, not the original paper subset or a held-out evaluation.\n'
        'HarmBench: MIT; refusal_direction: Apache-2.0; Alpaca data: CC BY-NC 4.0.\n'
        'Use for noncommercial research; preserve source attribution and local license notices.\n', encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'data/chat_refusal/downloads')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/calibration-inputs')
    args = parser.parse_args()
    print(prepare(args.directory, args.output))


if __name__ == '__main__':
    main()
