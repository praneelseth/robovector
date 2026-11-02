"""Load Arditi's pinned prompt pools; download only license-verified splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

REVISION = '9d852fae1a9121c78b29142de733cb1340770cc3'
BASE = f'https://raw.githubusercontent.com/andyrdt/refusal_direction/{REVISION}'
DEFAULT_DIR = Path(__file__).resolve().parent / 'downloads'
SPLITS = ('train', 'val', 'test')
LABELS = ('harmful', 'harmless')
LICENSE_URLS = {
    'refusal_direction_APACHE-2.0.txt': f'{BASE}/LICENSE',
    'HarmBench_MIT.txt': 'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/LICENSE',
    'JailbreakBench_MIT.txt': 'https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/LICENSE',
    'StrongREJECT_MIT.txt': 'https://raw.githubusercontent.com/alexandrasouly/strongreject/main/LICENSE',
    'Alpaca_CC-BY-NC-4.0.txt': 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/DATA_LICENSE',
}


def validate(harmtype, split):
    if harmtype not in LABELS or split not in SPLITS:
        raise ValueError('Expected harmful/harmless and train/val/test')


def download_split(harmtype: str, split: str, directory: Path = DEFAULT_DIR) -> Path:
    """Download exact published pool, never recreate or reshuffle its split."""
    validate(harmtype, split)
    if (harmtype, split) == ('harmful', 'train'):
        raise PermissionError('Mixed harmful_train includes MaliciousInstruct with unverified upstream license; see README.md. No download attempted.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    licenses = ['refusal_direction_APACHE-2.0.txt']
    licenses += ['Alpaca_CC-BY-NC-4.0.txt'] if harmtype == 'harmless' else ['HarmBench_MIT.txt']
    if (harmtype, split) == ('harmful', 'test'):
        licenses += ['JailbreakBench_MIT.txt', 'StrongREJECT_MIT.txt']
    for name in licenses:
        target = directory / name
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(LICENSE_URLS[name], timeout=60).read())
    name = f'{harmtype}_{split}.json'
    url = f'{BASE}/dataset/splits/{name}'
    target = directory / name
    raw = urllib.request.urlopen(url, timeout=60).read()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not all(isinstance(r.get('instruction'), str) for r in rows):
        raise ValueError('Unexpected upstream dataset schema')
    temp = target.with_suffix('.tmp')
    temp.write_bytes(raw)
    temp.replace(target)
    metadata = {'url': url, 'revision': REVISION, 'sha256': hashlib.sha256(raw).hexdigest(),
                'count': len(rows), 'licenses': {name: LICENSE_URLS[name] for name in licenses}}
    target.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return target


def load_split(harmtype: str, split: str, directory: Path = DEFAULT_DIR, *,
               download: bool = False, sample_size: int | None = None,
               seed: int = 42, instructions_only: bool = False) -> list:
    """Read local data; no implicit network. Sampling is optional and reproducible.

    These files are source pools, not necessarily the 128/32 prompts sampled by
    a particular paper run. Local harmful_train can be loaded after permission
    is independently resolved; this loader never downloads that mixed split.
    """
    validate(harmtype, split)
    path = Path(directory) / f'{harmtype}_{split}.json'
    if not path.exists() and download:
        download_split(harmtype, split, directory)
    rows = json.loads(path.read_text(encoding='utf-8'))
    if sample_size is not None:
        if not 0 < sample_size <= len(rows):
            raise ValueError('sample_size must be positive and no larger than pool')
        rows = random.Random(seed).sample(rows, sample_size)
    return [row['instruction'] for row in rows] if instructions_only else rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--harmtype', choices=LABELS)
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if bool(args.harmtype) != bool(args.split):
        parser.error('--harmtype and --split must be supplied together')
    selected = [(args.harmtype, args.split)] if args.harmtype else [
        (label, split) for label in LABELS for split in SPLITS if (label, split) != ('harmful', 'train')]
    for label, split in selected:
        rows = load_split(label, split, args.directory, download=args.download,
                          sample_size=args.sample_size, seed=args.seed)
        print(f'{label}/{split}: {len(rows)} instructions')


if __name__ == '__main__':
    main()
