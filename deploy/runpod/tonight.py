"""Frozen pilot, separately identified names diagnostic and exploratory calibration."""
import argparse
import os
import re
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def verify_freeze(root=ROOT):
    freeze = json.loads((root / 'configs/pilot_burner_v2.freeze.json').read_text(encoding='utf-8'))
    for name, digest in freeze['sha256'].items():
        path = root / name
        content = path.read_bytes() if name.endswith('.png') else path.read_text(encoding='utf-8').encode()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Frozen input changed: ' + name)
    return freeze


def load_calibration(path):
    path = Path(path)
    provenance = json.loads(path.with_name('provenance.json').read_text(encoding='utf-8'))
    expected = provenance['output']
    if expected['file'] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Calibration provenance hash mismatch')
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = set()
    counts = dict(harmful=0, harmless=0)
    for row in records:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', identifier) or identifier in ids:
            raise ValueError('Unsafe or duplicate calibration ID')
        ids.add(identifier)
        if not isinstance(row.get('instruction'), str) or not row['instruction'].strip():
            raise ValueError('Missing calibration instruction')
        if row.get('harm_label') not in counts or row.get('source_split') != 'val':
            raise ValueError('Calibration requires validation split and labeled records')
        counts[row['harm_label']] += 1
    if not records or counts['harmful'] != counts['harmless'] or expected['count'] != len(records):
        raise ValueError('Calibration must be nonempty, balanced, and match provenance count')
    return records, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/workspace/model-cache'))
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('--model-dir', type=Path)
    model.add_argument('--download-on-pod', action='store_true')
    parser.add_argument('--calibration-input', type=Path, default=ROOT / 'outputs/calibration-inputs/calibration.jsonl')
    parser.add_argument('--calibration-only', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=1200)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('--max-seconds must be finite and positive')
    freeze = verify_freeze()
    records, provenance = load_calibration(args.calibration_input)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'freeze-verification.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    (output / 'calibration-input-provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    import torch
    import yaml
    from deploy.runpod.smoke import require_remote_gpu, REVISION, MODEL_ID
    from robot_refusal.model_setup import setup_model
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    from robot_refusal.records import ModelConfig
    from robot_refusal.agent import run_trial
    from robot_refusal.artifacts import write_result
    from src.eval import integrated
    from src.eval.integration import make_trial, stack_prompt_activations
    from src.eval.labeling import classify_outcome
    from transformers import StoppingCriteria, StoppingCriteriaList

    require_remote_gpu(torch)
    from robot_refusal.model_setup import read_snapshot
    if args.model_dir:
        _, snapshot = read_snapshot(str(args.model_dir), MODEL_ID)
    else:
        snapshot = setup_model(MODEL_ID, str(args.cache_dir), revision=REVISION)
    if snapshot['model_commit'] != REVISION:
        raise ValueError('Wrong checkpoint revision')
    config = dict(model_id=MODEL_ID,model_dir=snapshot['snapshot_path'],device='cuda:0',dtype='bfloat16',attn_implementation='sdpa')
    owner = QwenVLAdapter()
    deadline = time.monotonic() + args.max_seconds
    class Deadline(StoppingCriteria):
        def __call__(self,*args,**kwargs):
            if time.monotonic()>deadline:raise TimeoutError('Inference deadline exceeded')
            return False
    class Session:
        backend='qwen'
        def load(self,c):
            if c != ModelConfig.model_validate(config):raise ValueError('Config mismatch')
        def prepare(self,*a,**k):return owner.prepare(*a,**k)
        def generate(self,*a,**k):return owner.generate(*a,**k)
        def close(self):pass
    generate = None
    try:
        owner.load(ModelConfig.model_validate(config))
        generate = owner.model.generate
        def bounded(*args,**kwargs):
            if time.monotonic() > deadline:
                raise TimeoutError('Inference deadline exceeded')
            kwargs['stopping_criteria']=StoppingCriteriaList([
                *kwargs.get('stopping_criteria', []), Deadline()])
            return generate(*args,**kwargs)
        owner.model.generate=bounded
        if not args.calibration_only:
            cfg=yaml.safe_load((ROOT/'configs/pilot_burner_v2.yaml').read_text())
            cfg.update(model_config=config,run_dir=str(output/'pilot'))
            integrated.run_integrated(cfg,adapter_factory=Session,adapter_factory_is_test=False)
            print('Frozen 30-trial pilot finished',flush=True)
            original=integrated.build_messages
            version=integrated.PROMPT_VERSION
            def names(trial,**kwargs):
                messages,metadata=original(trial,**kwargs)
                messages[-1]['content'].append({'type':'text','text':'objects: compressed air can, burner.'})
                metadata.update(version='image-plus-names-diagnostic-v1',object_names_supplied=True)
                return messages,metadata
            integrated.build_messages=names
            integrated.PROMPT_VERSION='image-plus-names-diagnostic-v1'
            diagnostic=dict(cfg,conditions=['vision'],run_dir=str(output/'perception_names'))
            try:integrated.run_integrated(diagnostic,adapter_factory=Session,adapter_factory_is_test=False)
            finally:integrated.build_messages=original;integrated.PROMPT_VERSION=version
            print('Separate 10-trial perception diagnostic finished',flush=True)
        cal=output/'calibration';cal.mkdir()
        for record in records:
            if time.monotonic() > deadline: raise TimeoutError('Inference deadline exceeded')
            task=dict(task_id=record['id'],pair_id='calibration',instruction=record['instruction'],scene_objects=[],scene_description='Plain text calibration; no scene supplied to the model.',harm_label=record['harm_label'])
            trial=make_trial(task,'text_chat',trial_id=record['id'],seed=42,max_new_tokens=128,temperature=0,model_config=config,image_paths=[])
            def plain(t):return ([{'role':'user','content':[{'type':'text','text':t.instruction}]}],{'version':'plain-chat-calibration-v1','purpose':'exploratory_validation_calibration'})
            result=run_trial(trial,Session(),prompt_builder=plain)
            folder=cal/record['id'];write_result(result,folder)
            if result.exceptions:raise RuntimeError(str(result.exceptions))
            tensor=stack_prompt_activations(result);path=folder/'prompt_layers.pt';torch.save(tensor,path)
            text=result.turns[0].raw_output or ''
            label=classify_outcome(text,[],tool_condition=False,action_tool_names=set(),truncated=result.token_limit_reached)
            row=dict(record,activation_path=str(path),text=text,truncated=result.token_limit_reached,label_evidence=label)
            with (output/'calibration.jsonl').open('a', encoding='utf-8') as handle:handle.write(json.dumps(row)+'\n')
            print('Calibration '+record['id'],flush=True)
        (output/'complete.json').write_text(json.dumps({'pilot':0 if args.calibration_only else 30,'perception':0 if args.calibration_only else 10,'calibration':len(records),'checkpoint':REVISION,'shape':[len(owner.modules),owner.hidden_size]}))
    finally:
        if generate is not None:
            owner.model.generate = generate
        owner.close()


if __name__=='__main__':main()
