"""Run the fixed Qwen capture/evaluation smoke on a Runpod GPU, never locally.

Reuses the prepared GPU harness and partner adapter. Runpod lifecycle is managed
externally: an exited Python process does NOT stop GPU billing.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'


def require_remote_gpu(torch):
    if platform.system() != 'Linux' or not os.environ.get('RUNPOD_POD_ID'):
        raise RuntimeError('Run only inside the intended Runpod Pod; downloads on the laptop are prohibited')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one CUDA GPU')
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
        raise RuntimeError('Use a 48 GB or larger GPU for this unquantized capture check')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 support required')


def main():
    # Runpod images may enable an optional downloader absent from our pinned venv.
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/workspace/model-cache'))
    parser.add_argument('--download-on-pod', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    import torch
    require_remote_gpu(torch)  # Must precede every possible weight download.
    from robot_refusal.model_setup import setup_model, read_snapshot
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.records import ModelConfig
    from deploy.qwen_smoke.model.model import Model

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hardware = dict(pod_id=os.environ['RUNPOD_POD_ID'], gpu=torch.cuda.get_device_name(0),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__, python=platform.python_version(), checkpoint=REVISION)
    (output / 'hardware.json').write_text(json.dumps(hardware, indent=2))
    if args.model_dir:
        _, snapshot = read_snapshot(str(args.model_dir), MODEL_ID)
    elif args.download_on_pod:
        snapshot = setup_model(MODEL_ID, str(args.cache_dir), revision=REVISION)
    else:
        raise ValueError('Supply --model-dir or explicitly permit --download-on-pod')
    if snapshot['model_commit'] != REVISION:
        raise ValueError('Wrong checkpoint revision')
    model_config = dict(model_id=MODEL_ID, model_dir=snapshot['snapshot_path'],
                        device='cuda:0', dtype='bfloat16', attn_implementation='sdpa')
    adapter = QwenVLAdapter()
    adapter.load(ModelConfig.model_validate(model_config))
    harness = Model(data_dir=ROOT / 'data')
    harness.root, harness.model_config, harness.adapter = output, model_config, adapter
    try:
        result = harness.predict({'operation': 'smoke-v1'})
        (output / 'artifacts.zip').write_bytes(base64.b64decode(result['archive_base64']))
        (output / 'status.json').write_text(json.dumps(result['status'], indent=2))
        if result['status']['error']:
            raise RuntimeError(result['status']['error'])
        rows = [json.loads(s) for s in (output / 'artifacts/run/results.jsonl').read_text().splitlines()]
        expected = [len(adapter.modules), adapter.hidden_size]
        bad = [r for r in rows if r.get('error') or r['truncated'] or not r['text'].strip()
               or r['activation_shape'] != expected]
        if len(rows) != 12 or bad:
            raise RuntimeError(f'Incomplete smoke: {len(rows)} rows, {len(bad)} failed or pending rows')
        for row in rows:
            tensor = torch.load(row['activation_path'], weights_only=True, map_location='cpu')
            if list(tensor.shape) != expected or not torch.isfinite(tensor).all():
                raise RuntimeError('Invalid saved activation tensor')
        print(json.dumps(dict(passed=True, trials=len(rows), activation_shape=expected,
                              artifact=str(output / 'artifacts.zip')), indent=2), flush=True)
    finally:
        adapter.close()


if __name__ == '__main__':
    main()
