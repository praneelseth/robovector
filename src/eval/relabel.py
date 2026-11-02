"""Relabel saved paired-eval outputs without model calls or source mutations."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.eval.labeling import LABELER_VERSION, classify_outcome
from src.eval.run_eval import summarize, write_jsonl


def relabel_run(source: Path, destination: Path, *, action_tool_names=None) -> Path:
    """Create a separate derived result set with provenance and previous labels."""
    source, destination = Path(source), Path(destination)
    source_results = source / 'results.jsonl'
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if action_tool_names is None:
        action_tool_names = source_manifest.get('config', {}).get('action_tool_names')
        if action_tool_names is None:
            raise ValueError('Source has no action_tool_names; provide them explicitly')
    raw = source_results.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    updated = []
    for row in rows:
        record = dict(row)
        record['source_labels'] = {key: record.pop(key) for key in
                                  ('outcome', 'rule_outcome', 'judge_outcome', 'labeler_version', 'label_evidence') if key in record}
        error = record.get('error')
        if record['source_labels'].get('outcome') == 'error' and not error:
            error = 'Source record was labeled error; retained during offline relabeling'
        evidence = classify_outcome(record['text'], record['tool_calls'],
                                    tool_condition=record['condition'] in ('text_tools', 'vision_tools'),
                                    action_tool_names=set(action_tool_names), error=error)
        record.update(outcome=evidence['outcome'], rule_outcome=evidence['outcome'],
                      label_evidence=evidence, labeler_version=LABELER_VERSION,
                      labeling_method='offline_rules', source_results=str(source_results.resolve()))
        updated.append(record)
    # Never overwrite a prior run or derived artifact.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(artifact_type='offline_relabel', labeler_version=LABELER_VERSION,
                    source_run=str(source.resolve()), source_results_sha256=hashlib.sha256(raw).hexdigest(),
                    source_labeler_version=source_manifest.get('labeler_version'),
                    action_tool_names=list(action_tool_names), model_calls=0,
                    generated_at=datetime.now(timezone.utc).isoformat())
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    write_jsonl(destination / 'results.jsonl', updated)
    summarize(updated, destination / 'summary.csv')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--action-tool-names', nargs='+')
    args = parser.parse_args()
    output = args.output_dir or Path('outputs/relabels') / (
        args.source_run.name + '-' + LABELER_VERSION + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    print(relabel_run(args.source_run, output, action_tool_names=args.action_tool_names))


if __name__ == '__main__':
    main()
