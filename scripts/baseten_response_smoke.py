"""Temporary response-only API check; never claims activation or Qwen validation.

Reuses integrated prompts, strict JSON parser, label inputs and summaries. No
robot commands are executed. One writer only; ambiguous requests are not retried.
"""
import argparse
import base64
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import requests
from scripts.baseten_smoke import credential, save
from src.eval.integration import make_trial, build_messages, label_inputs, ACTION_NAMES
from src.eval.labeling import classify_outcome
from src.eval.run_eval import summarize
from robot_refusal.tools import parse_response

MODEL = 'zai-org/GLM-5.3-Flash'
BASE = 'https://inference.baseten.co/v1'


def score(raw, trial, condition, finish_reason, native_calls=None):
    parsed = parse_response(raw, trial.object_manifest, 0, trial.max_total_commands)
    turn = SimpleNamespace(raw_output=raw, **parsed)
    text, calls, invalid = label_inputs(SimpleNamespace(turns=[turn], condition=trial.condition))
    if condition != 'text_tools':
        text, calls, invalid = raw, [], False
    invalid |= bool(native_calls)
    label = classify_outcome(text, calls, tool_condition=condition == 'text_tools',
        action_tool_names=ACTION_NAMES, invalid_action=invalid,
        truncated=finish_reason == 'length', error='Empty output' if not raw.strip() and finish_reason != 'length' else None)
    return dict(**label, assistant_text=text, tool_calls=calls,
                validation_errors=parsed['validation_errors'] if condition == 'text_tools' else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    key = credential(args.env_file)
    headers = {'Authorization': 'Bearer ' + key}
    catalog = requests.get(BASE + '/models', headers=headers, timeout=30)
    catalog.raise_for_status()
    selected = next(m for m in catalog.json()['data'] if m['id'] == MODEL)
    pricing = selected['pricing']
    # Conservative bound uses the entire advertised context on every request,
    # including image tokens, even though actual prompts are much shorter.
    bound = 24 * (selected['context_length'] * float(pricing['prompt']) +
                  1024 * float(pricing['completion']) + float(pricing['request']))
    if bound > 4 or float(pricing['image']) != 0:
        raise ValueError('Catalog pricing exceeds conservative $4 sub-budget')
    tasks = [json.loads(line) for line in (ROOT / 'data/tasks.jsonl').read_text().splitlines()]
    tasks = [t for t in tasks if t['task_id'] in ('burner', 'burner_safe')]
    assert len(tasks) == 2 and all(t['review_status'] == 'approved' for t in tasks)
    image = ROOT / 'data/images/burner_real.png'
    manifest = dict(engine='response-only-api-smoke-v1', model=MODEL, tasks=tasks,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), catalog=selected,
        activation_status='unavailable_from_hosted_api', n_repeats=2,
        conditions=['text_chat', 'text_tools', 'vision'], temperature=0.7,
        reasoning_effort='none', token_budgets=[256, 1024], seed=None,
        maximum_requests=24, conservative_cost_bound_usd=bound)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Manifest changed; refusing to mix runs')
    else:
        save(manifest_path, manifest)
    records, cost = [], 0.0
    for task, condition, repeat in itertools.product(tasks, manifest['conditions'], range(2)):
        name = f"{task['task_id']}_{condition}_{repeat}"
        # ModelConfig is Qwen-specific; this object is used ONLY for shared prompt
        # construction. It is never executed or serialized as the API model identity.
        trial = make_trial(task, condition, trial_id=name, seed=42, max_new_tokens=256,
            temperature=0.7, model_config={}, image_paths=[str(image)] if condition == 'vision' else [])
        messages, metadata = build_messages(trial)
        for message in messages:
            for part in message['content']:
                if part['type'] == 'image':
                    part.clear()
                    part.update(type='image_url', image_url={'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()})
        attempts = []
        for budget in manifest['token_budgets']:
            stem = folder / f'{name}_{budget}'
            response_path = stem.with_suffix('.response.json')
            request_path = stem.with_suffix('.request.json')
            payload = dict(model=MODEL, messages=messages, temperature=0.7,
                           max_tokens=budget, reasoning_effort='none')
            if response_path.exists():
                response = json.loads(response_path.read_text())
            else:
                if request_path.exists():
                    raise RuntimeError('Ambiguous previous request; inspect before retrying: ' + name)
                save(request_path, payload)
                response_http = requests.post(BASE + '/chat/completions', headers=headers, json=payload, timeout=(10, 90))
                if not response_http.ok:
                    save(stem.with_suffix('.error.json'), {'status': response_http.status_code, 'body': response_http.text})
                    response_http.raise_for_status()
                response = response_http.json()
                save(response_path, response)
            choice = response['choices'][0]
            raw = choice['message'].get('content') or ''
            usage = response['usage']
            estimate = usage['prompt_tokens'] * float(pricing['prompt']) + usage['completion_tokens'] * float(pricing['completion'])
            cost += estimate
            attempts.append(dict(response_file=str(response_path), max_tokens=budget, usage=usage,
                                 finish_reason=choice['finish_reason'], estimated_cost_usd=estimate))
            scored = score(raw, trial, condition, choice['finish_reason'], choice['message'].get('tool_calls'))
            if not scored['truncated']:
                break
        row = dict(model=MODEL, task_id=task['task_id'], harm_label=task['harm_label'],
            condition=condition, repeat=repeat, text=raw, attempts=attempts, prompt_metadata=metadata,
            activation_path=None, activation_shape=None, activation_status='unavailable_from_hosted_api', **scored)
        records.append(row)
        (folder / 'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
        summarize(records, folder / 'summary.csv')
        print(json.dumps({k: row[k] for k in ('task_id', 'condition', 'repeat', 'outcome', 'truncated')}), flush=True)
    save(folder / 'cost.json', dict(estimated_cost_usd=cost, pricing_source='live Baseten /v1/models',
         basis='reported tokens at uncached list prices; not an invoice', requests=sum(len(r['attempts']) for r in records)))
    print(f'Completed {len(records)} trials; estimated token cost ${cost:.6f}', flush=True)


if __name__ == '__main__':
    main()
