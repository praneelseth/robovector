"""Build a small explicit-allowlist upload; excludes credentials and weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(destination):
    files = {}
    for folder in ('src/eval', 'src/stubs', 'src/robot_refusal'):
        for path in (ROOT / folder).rglob('*.py'):
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ('src/__init__.py', 'src/interfaces.py', 'data/__init__.py', 'data/prompt_templates.py',
                 'data/images/burner_real.png', 'deploy/qwen_smoke/model/model.py',
                 'deploy/runpod/smoke.py', 'deploy/runpod/requirements.txt', 'deploy/runpod/README.md'):
        files[name] = (ROOT / name).read_bytes()
    tasks = [json.loads(s) for s in (ROOT / 'data/tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks = [t for t in tasks if t['task_id'] in ('burner', 'burner_safe')]
    if len(tasks) != 2 or any(t.get('review_status') != 'approved' for t in tasks):
        raise ValueError('Expected approved burner pair')
    files['data/tasks.jsonl'] = ''.join(json.dumps(t)+'\n' for t in tasks).encode()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr('bundle-manifest.json', json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(f'Packaged {len(package(args.destination))} files to {args.destination}')
