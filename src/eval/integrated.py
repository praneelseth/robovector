"""Resumable paired evaluation consuming the single-trial runner's artifacts."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from functools import partial
from pathlib import Path

import torch

from robot_refusal.agent import run_trial
from robot_refusal.artifacts import write_result
from robot_refusal.model_setup import read_snapshot
from robot_refusal.models.qwen_vl import QwenVLAdapter
from robot_refusal.tools import parse_response
from src.eval.integration import (ACTION_NAMES, CONDITION_MAP, PROMPT_VERSION, build_messages,
                                 label_inputs, make_trial, stack_prompt_activations, parse_visual_descriptions)
from src.eval.labeling import LABELER_VERSION, classify_outcome


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_integrated(config, *, adapter_factory=None):
    from src.eval.run_eval import digest, read_jsonl, summarize, write_jsonl

    cfg = dict(tasks='data/tasks.jsonl', task_ids=['burner', 'burner_safe'],
        conditions=list(CONDITION_MAP), n_repeats=2, temperature=0.7, seed=42,
        max_new_tokens=256, truncation_max_new_tokens=2048,
        image_dir='data/images', image_source='real', output_dir='outputs',
        model_config={'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'device': 'cuda:0'})
    cfg.update(config)
    if not cfg['conditions'] or not set(cfg['conditions']) <= CONDITION_MAP.keys():
        raise ValueError('Unknown or empty conditions')
    if len(set(cfg['conditions'])) != len(cfg['conditions']):
        raise ValueError('Duplicate conditions')
    if cfg['n_repeats'] < 1 or cfg['max_new_tokens'] < 1 or cfg['temperature'] < 0:
        raise ValueError('Invalid generation configuration')
    if cfg['truncation_max_new_tokens'] <= cfg['max_new_tokens']:
        raise ValueError('Truncation retry cap must exceed the initial token budget')
    if cfg['image_source'] not in ('real', 'sim'):
        raise ValueError('Unknown image source')
    if cfg.get('judge'):
        raise ValueError('Integrated v1 uses audited rule labels; judge override is disabled')
    if 'vision_tools' in cfg['conditions'] and cfg.get('vision_tool_grounding') != 'model_descriptions':
        raise ValueError('Vision tool grounding unresolved: explicitly choose model_descriptions or omit vision_tools')
    tasks = read_jsonl(Path(cfg['tasks']))
    selected = [task for task in tasks if task['task_id'] in cfg['task_ids']]
    if len({t['task_id'] for t in selected}) != len(cfg['task_ids']) or len(selected) != len(cfg['task_ids']):
        raise ValueError('Missing or duplicate selected tasks')
    if any(t.get('review_status') != 'approved' or t.get('excluded_from_v1') for t in selected):
        raise ValueError('Integrated v1 requires approved, non-excluded tasks')
    is_test = adapter_factory is not None
    if not is_test:
        # Reads a previously downloaded snapshot only. No network/download path.
        _, snapshot = read_snapshot(cfg['model_config'].get('model_dir'), cfg['model_config']['model_id'])
        adapter_factory = QwenVLAdapter
    else:
        snapshot = {'test_adapter': True, 'factory': adapter_factory.__module__ + '.' + adapter_factory.__qualname__}
    images, image_hashes = {}, {}
    if any(c.startswith('vision') for c in cfg['conditions']):
        for task in selected:
            ids = [key for key in task['image_ids'] if key.endswith('_' + cfg['image_source'])]
            if len(ids) != 1 or Path(ids[0]).name != ids[0] or any(c in ids[0] for c in '/\\'):
                raise ValueError('Expected one safe image ID per selected task')
            path = (Path(cfg['image_dir']) / (ids[0] + '.png')).resolve()
            images[task['task_id']] = [str(path)]
            image_hashes[str(path)] = file_hash(path)
    manifest = dict(config={k: v for k, v in cfg.items() if k != 'run_dir'}, tasks=selected,
        snapshot=snapshot, image_sha256=image_hashes, prompt_version=PROMPT_VERSION,
        labeler_version=LABELER_VERSION, engine='integrated-first-response-v1',
        action_tool_names=sorted(ACTION_NAMES), is_test=is_test)
    fingerprint = digest(manifest)
    run_dir = Path(cfg.get('run_dir') or Path(cfg['output_dir']) / 'runs' /
                   datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    run_dir.mkdir(parents=True, exist_ok=True)
    mp = run_dir / 'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Resume configuration/data differ; use a new run directory')
    else:
        mp.write_text(json.dumps(dict(fingerprint=fingerprint, **manifest), indent=2), encoding='utf-8')
    result_path = run_dir / 'results.jsonl'
    records = {r['run_key']: r for r in read_jsonl(result_path, recover_trailing=True)}

    def persist():
        write_jsonl(result_path, list(records.values()))
        write_jsonl(run_dir / 'activations.jsonl', [
            {k: row[k] for k in ('run_key', 'model', 'task_id', 'condition', 'repeat', 'seed',
                               'activation_path', 'activation_shape', 'activation_sha256')}
            for row in records.values() if row.get('activation_path')])
        summarize(list(records.values()), run_dir / 'summary.csv')

    for task, condition, repeat in itertools.product(selected, cfg['conditions'], range(cfg['n_repeats'])):
        key = digest([fingerprint, task['task_id'], condition, repeat])
        old = records.get(key)
        if old:
            if all(Path(path).is_file() and file_hash(path) == sha for path, sha in old['artifact_hashes'].items()):
                continue
            raise ValueError('Saved artifacts missing or changed; use a new run directory, preserving evidence')
        seed = (cfg['seed'] + int(digest([cfg['model_config']['model_id'], task['task_id'], condition, repeat])[:8], 16)) % 2**32
        row = dict(run_key=key, model=cfg['model_config']['model_id'], task_id=task['task_id'],
            pair_id=task['pair_id'], harm_label=task['harm_label'], twin_type=task.get('twin_type'),
            condition=condition, repeat=repeat, seed=seed, is_test=is_test,
            labeler_version=LABELER_VERSION, outcome='error', error=None, truncated=False,
            text='', tool_calls=[], attempts=[], activation_path=None, artifact_hashes={})
        budget = cfg['max_new_tokens']
        while True:
            # Each attempt is immutable. A crash before the JSONL checkpoint leaves
            # artifacts that are loaded on resume rather than silently regenerated.
            attempt_dir = run_dir / 'trials' / key / str(budget)
            trial = make_trial(task, condition, trial_id=key, seed=seed,
                max_new_tokens=budget, temperature=cfg['temperature'],
                model_config=cfg['model_config'], image_paths=images.get(task['task_id'], []) if condition.startswith('vision') else [])
            saved = attempt_dir / 'result.json'
            if saved.exists():
                from robot_refusal.records import AgentTrialResult
                result = AgentTrialResult.model_validate_json(saved.read_text(encoding='utf-8'))
                result._activation_tensors = torch.load(attempt_dir / 'activations.pt', weights_only=True)
            else:
                result = run_trial(trial, adapter_factory(),
                    prompt_builder=partial(build_messages, vision_tool_grounding=cfg.get('vision_tool_grounding')),
                    response_parser=parse_visual_descriptions if condition == 'vision_tools' else parse_response)
                write_result(result, attempt_dir)
            for path in attempt_dir.iterdir():
                if path.is_file():
                    row['artifact_hashes'][str(path.resolve())] = file_hash(path)
            row['attempts'].append(dict(max_new_tokens=budget, truncated=result.token_limit_reached,
                trial_path=str(saved.resolve()), termination_reason=result.termination_reason))
            row['truncated'] = result.token_limit_reached
            row['object_grounding'] = result.prompt_treatment.get('object_grounding')
            row['max_new_tokens'] = budget
            if result.turns:
                turn = result.turns[0]
                row.update(text=turn.raw_output or '', messages=turn.messages,
                           assistant_text=turn.assistant_text, validation_errors=turn.validation_errors)
                text, calls, invalid = label_inputs(result)
                row.update(tool_calls=calls, invalid_action=invalid,
                           call_records=[call.model_dump() for call in turn.calls])
            else:
                text, calls, invalid = '', [], False
            if result.exceptions:
                row['error'] = json.dumps(result.exceptions)
            else:
                try:
                    hidden = stack_prompt_activations(result)
                    activation = attempt_dir / 'prompt_layers.pt'
                    if not activation.exists():
                        torch.save(hidden, activation)
                    row.update(activation_path=str(activation.resolve()), activation_shape=list(hidden.shape),
                               activation_sha256=file_hash(activation))
                    row['artifact_hashes'][str(activation.resolve())] = file_hash(activation)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
            if row['truncated'] and not row['error'] and budget < cfg['truncation_max_new_tokens']:
                budget = min(budget * 2, cfg['truncation_max_new_tokens'])
                continue
            evidence = classify_outcome(text, calls, tool_condition=condition.endswith('tools'),
                action_tool_names=ACTION_NAMES, invalid_action=invalid,
                truncated=row['truncated'], error=row['error'])
            row.update(outcome=evidence['outcome'], rule_outcome=evidence['outcome'], label_evidence=evidence)
            records[key] = row
            persist()
            break
    persist()
    return run_dir
