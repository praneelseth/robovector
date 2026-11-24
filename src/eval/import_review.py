"""Import human annotations without overwriting source labels or adjudicating them."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from src.eval.labeling import LABELS, classify_outcome


def parse_annotations(text):
    annotations = {}
    for identifier, body in re.findall(r'^## (R\d+)\b[^\n]*\n(.*?)(?=^## R\d+\b|\Z)', text, re.M | re.S):
        if identifier in annotations:
            raise ValueError('Duplicate review ID: ' + identifier)
        label = re.search(r'Human label:\s*\*\*\[(.*?)\]\*\*', body)
        notes = re.search(r'Notes:\s*\[(.*?)\](?=\s*\n)', body, re.S)
        if not label or label.group(1).strip() not in LABELS:
            raise ValueError('Missing or unknown human label: ' + identifier)
        annotations[identifier] = dict(human_label=label.group(1).strip(),
                                       notes=notes.group(1) if notes else '')
    if not annotations:
        raise ValueError('No review annotations found')
    return annotations


def import_review(review, packet, output):
    review, packet, output = map(Path, (review, packet, output))
    annotations = parse_annotations(review.read_text(encoding='utf-8-sig'))
    rows = [json.loads(line) for line in packet.read_text(encoding='utf-8').splitlines() if line.strip()]
    identifiers = [row['review_id'] for row in rows]
    if len(set(identifiers)) != len(rows) or set(identifiers) != set(annotations):
        raise ValueError('Annotation IDs must match the original packet exactly')
    records = []
    for row in rows:
        evidence = classify_outcome(row['text'] if not row.get('validated_calls') else
            _assistant_text(row['text']), row.get('validated_calls', []),
            tool_condition=row['condition'].endswith('tools'),
            action_tool_names={'pick', 'place', 'pour', 'move_held_object', 'release'},
            invalid_action=bool(row.get('validation_errors') or
                any(not call.get('valid') for call in row.get('call_records', []))))
        annotation = annotations[row['review_id']]
        records.append(dict(row, **annotation, reviewer='user', automatic_evidence=evidence,
            disagrees=annotation['human_label'] != evidence['outcome'],
            adjudicated_label=None, adjudication_status='pending'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'annotations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    manifest = dict(n=len(records), disagreements=sum(r['disagrees'] for r in records),
        interpretation='Diagnostic sample; agreement is not a held-out accuracy estimate.',
        review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
        packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
        labeler_source_sha256=hashlib.sha256(Path(__file__).with_name('labeling.py').read_bytes()).hexdigest(), model_calls=0)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def _assistant_text(raw):
    try:
        envelope = json.loads(raw)
        return envelope.get('assistant_text', raw) if isinstance(envelope, dict) else raw
    except ValueError:
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_review(args.review, args.packet, args.output)))


if __name__ == '__main__':
    main()
