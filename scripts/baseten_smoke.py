"""Prepare, deploy, monitor and stop only the bounded qwen-smoke deployment.

Credentials are read in memory from the explicitly selected existing env file.
This script never writes credentials, runs other projects, or calls shared APIs.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAME = 'qwen-smoke'
MAX_WALL_SECONDS = 30 * 60


def credential(path):
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        name, sep, value = line.strip().partition('=')
        if sep and name == 'BASETEN_API_KEY':
            return value.strip().strip('"').strip("'")
    raise ValueError('Baseten key not found')


def api(key, path, method='GET', payload=None):
    response = requests.request(method, 'https://api.baseten.co/v1' + path,
        headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=(10, 30))
    if not response.ok:
        raise RuntimeError(f'Baseten management HTTP {response.status_code}: {response.text[:500]}')
    return response.json()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def prepare(bundle):
    if bundle.exists():
        raise FileExistsError('Use a fresh bundle directory')
    bundle.mkdir(parents=True)
    checkpoint = json.loads((ROOT / 'deploy/qwen_smoke/checkpoint.json').read_text())
    (bundle / 'model').mkdir()
    shutil.copy2(ROOT / 'deploy/qwen_smoke/model/model.py', bundle / 'model/model.py')
    packages = bundle / 'packages'
    for folder in ('src/eval', 'src/stubs', 'src/robot_refusal'):
        destination = packages / ('robot_refusal' if folder == 'src/robot_refusal' else folder)
        shutil.copytree(ROOT / folder, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (packages / 'data').mkdir(parents=True)
    for name in ('data/__init__.py', 'data/prompt_templates.py', 'src/__init__.py', 'src/interfaces.py'):
        shutil.copy2(ROOT / name, packages / name)
    (bundle / 'data/images').mkdir(parents=True)
    tasks = [json.loads(line) for line in (ROOT / 'data/tasks.jsonl').read_text().splitlines()]
    selected = [task for task in tasks if task['task_id'] in ('burner', 'burner_safe')]
    (bundle / 'data/tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in selected))
    shutil.copy2(ROOT / 'data/images/burner_real.png', bundle / 'data/images/burner_real.png')
    save(bundle / 'data/checkpoint.json', checkpoint)
    config = dict(model_name=NAME, python_version='py311',
        requirements=['torch==2.7.1', 'torchvision==0.22.1', 'transformers==4.57.1',
            'Pillow==11.3.0', 'pydantic==2.12.3', 'huggingface-hub==0.34.3',
            'accelerate==1.9.0', 'safetensors==0.5.3', 'numpy==2.2.6', 'PyYAML==6.0.3'],
        resources={'accelerator': 'A100', 'use_gpu': True, 'cpu': '12', 'memory': '144Gi'},
        runtime={'predict_concurrency': 1},
        weights=[{'source': 'hf://Qwen/Qwen3-VL-8B-Instruct@' + checkpoint['sha'],
                  'mount_location': '/models/qwen',
                  'allow_patterns': ['*.json', '*.safetensors', '*.txt', '*.jinja']}])
    (bundle / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    files = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in bundle.rglob('*') if p.is_file()}
    save(bundle.parent / 'bundle-manifest.json', {'files': files,
        'bytes': sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())})
    print(f'Prepared {len(files)} allowlisted files; no secrets or weights in upload', flush=True)


def deployment_path(state):
    return f"/models/{state['model_id']}/deployments/{state['deployment_id']}"


def discover(key, state):
    models = api(key, '/models?name=' + NAME)['models']
    if len(models) > 1:
        raise RuntimeError('Ambiguous model name; refusing to modify deployments')
    if not models:
        return []
    model = models[0]
    items = api(key, f"/models/{model['id']}/deployments")['deployments']
    # Only deployments created after this explicitly authorized session.
    return [dict(model_id=model['id'], deployment_id=item['id']) for item in items
            if datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')).timestamp() >= state['started_at'] - 5]


def guard(key, state_path):
    """Independent cutoff; continues even if the interactive agent turn ends."""
    while True:
        state = json.loads(state_path.read_text())
        save(state_path.parent / 'guard-heartbeat.json', {'time': time.time(), 'pid': os.getpid()})
        if state.get('shutdown_confirmed'):
            return
        if time.time() >= state['deadline'] or state.get('stop_requested'):
            try:
                targets = ([state] if state.get('deployment_id') else discover(key, state))
                if not targets and state.get('push_finished'):
                    return
                for target in targets:
                    api(key, deployment_path(target) + '/deactivate', 'POST', {})
                    observed = api(key, deployment_path(target))
                    save(state_path.parent / 'guard-last-status.json', observed)
                    if observed['active_replica_count'] == 0 and observed['status'] in ('SCALED_TO_ZERO', 'INACTIVE', 'DEACTIVATED', 'BUILD_FAILED', 'DEPLOY_FAILED', 'FAILED'):
                        save(state_path.parent / 'guard-stopped.json', {'time': time.time(), 'status': observed})
                        return
            except Exception as exc:
                save(state_path.parent / 'guard-error.json', {'time': time.time(), 'error': str(exc)})
        time.sleep(20)


def deploy(key, env_file, folder):
    from truss import load
    from truss.remote.baseten.remote import BasetenRemote
    if api(key, '/organizations/me').get('name') != 'TraceAI':
        raise RuntimeError('Wrong workspace')
    if api(key, '/models?name=' + NAME)['models']:
        raise RuntimeError('qwen-smoke already exists; inspect rather than create another')
    state_path = folder / 'state.json'
    state = dict(started_at=time.time(), deadline=time.time()+MAX_WALL_SECONDS,
                 model_name=NAME, budget_usd=5, published_rate_usd_per_hour=4.0002,
                 max_wall_seconds=MAX_WALL_SECONDS, expected_max_compute_usd=2.01)
    save(state_path, state)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with (folder / 'guard.log').open('w') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard',
            '--folder', str(folder), '--env-file', str(env_file)], stdout=log, stderr=log,
            creationflags=flags, start_new_session=os.name != 'nt')
    for _ in range(20):
        heartbeat = folder / 'guard-heartbeat.json'
        if heartbeat.exists() and json.loads(heartbeat.read_text())['time'] >= state['started_at']:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError('Watchdog did not start; no deployment created')
    state['guard_pid'] = child.pid
    save(state_path, state)
    remote = BasetenRemote('https://app.baseten.co', api_key=key, api_key_use_bearer=True)
    try:
        service = remote.push(load(folder / 'bundle'), NAME, working_dir=folder,
            publish=True, promote=False, deployment_name='smoke', include_git_info=False,
            deploy_timeout_minutes=15)
        state.update(model_id=service.model_id, deployment_id=service.model_version_id,
                     predict_url=service.predict_url, push_finished=True)
        save(state_path, state)
        settings = api(key, deployment_path(state)+'/autoscaling_settings', 'PATCH',
            {'min_replica': 0, 'max_replica': 1, 'scale_down_delay': 60, 'concurrency_target': 1})
        save(folder / 'autoscaling-update.json', settings)
        print(json.dumps(state), flush=True)
    except Exception:
        state.update(push_finished=True, stop_requested=True)
        save(state_path, state)
        raise


def invoke(key, folder):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text())
    if time.time() + 600 > state['deadline']:
        raise RuntimeError('Not enough budget time remaining for inference')
    deployment = api(key, deployment_path(state))
    settings = deployment.get('autoscaling_settings') or {}
    if settings.get('max_replica') != 1 or settings.get('min_replica') != 0:
        raise RuntimeError('Replica controls not verified')
    if deployment.get('instance_type_name') != 'A100:12x144':
        raise RuntimeError('Unexpected GPU allocation')
    attempt = folder / 'inference-requested.json'
    if attempt.exists():
        raise RuntimeError('Inference already requested; inspect before retrying')
    save(attempt, {'time': time.time(), 'operation': 'smoke-v1'})
    try:
        response = requests.post(state['predict_url'], headers={'Authorization': 'Bearer '+key},
            json={'operation': 'smoke-v1'}, timeout=(15, 600))
        response.raise_for_status()
        payload = response.json()
        (folder / 'artifacts.zip').write_bytes(base64.b64decode(payload.pop('archive_base64'), validate=True))
        save(folder / 'smoke-status.json', payload)
        destination = folder / 'artifacts'
        with zipfile.ZipFile(folder / 'artifacts.zip') as archive:
            for item in archive.infolist():
                path = (destination / item.filename).resolve()
                if not path.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe artifact path')
            archive.extractall(destination)
        print(json.dumps(payload), flush=True)
    finally:
        state['stop_requested'] = True
        save(state_path, state)
        api(key, deployment_path(state)+'/deactivate', 'POST', {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'deploy', 'guard', 'status', 'invoke', 'down'])
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--env-file', type=Path)
    args = p.parse_args()
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(folder / 'bundle')
        return
    key = credential(args.env_file)
    if args.action == 'deploy':
        deploy(key, args.env_file.resolve(), folder)
    elif args.action == 'guard':
        guard(key, folder / 'state.json')
    elif args.action == 'invoke':
        invoke(key, folder)
    else:
        state = json.loads((folder / 'state.json').read_text())
        if args.action == 'down':
            state['stop_requested'] = True
            save(folder / 'state.json', state)
            api(key, deployment_path(state)+'/deactivate', 'POST', {})
        result = api(key, deployment_path(state))
        save(folder / 'last-status.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
