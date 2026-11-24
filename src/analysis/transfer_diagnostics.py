"""CPU-only follow-up of saved five-pair probes; no refitting or threshold tuning.

Only load trusted, locally generated joblib files. Original artifacts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .grouped_probe import validate_rows
from .refusal_plot import load_activations, read_rows


def score_summary(labels, scores):
    """Use unsquashed margins for ranking: sigmoid probabilities can saturate."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Require both labels and finite scores')
    predicted = (scores > 0).astype(int)
    return dict(balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                margin_auc=float(roc_auc_score(labels, scores)),
                predicted_harmful=int(predicted.sum()), n=len(labels),
                harmful_mean_margin=float(scores[labels == 1].mean()),
                safe_mean_margin=float(scores[labels == 0].mean()),
                min_margin=float(scores.min()), max_margin=float(scores.max()))


def verify_training_split(bundle, rows, held_pair, train_condition):
    expected = {r['run_key'] for r in rows
                if r['pair_id'] != held_pair and r['condition'] == train_condition}
    split = bundle['split']
    if (set(split['train_run_keys']) != expected or split['held_out_pair'] != held_pair
            or split['train_condition'] != train_condition):
        raise ValueError('Saved model training split does not match pair-held-out design')


def analyze(results, artifact_root, probes, output):
    import joblib
    output, probes = Path(output), Path(probes)
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    rows = read_rows(results)
    pairs = validate_rows(rows)
    digest = hashlib.sha256(Path(results).read_bytes()).hexdigest()
    metadata = json.loads((probes / 'metadata.json').read_text())
    if metadata['results_sha256'] != digest or metadata['status'] != 'complete':
        raise ValueError('Probe source hash/status mismatch')
    acts = load_activations(rows, results, artifact_root).numpy()
    labels = np.array([int(r['harm_label'] == 'harmful') for r in rows])
    output.mkdir(parents=True)
    metrics, predictions, model_hashes = [], [], {}
    for fold, pair in enumerate(pairs):
        for train in ('text_chat', 'text_tools'):
            for layer in range(acts.shape[1]):
                path = probes / 'models' / f'fold-{fold}_{train}_layer-{layer:02d}.joblib'
                bundle = joblib.load(path)
                verify_training_split(bundle, rows, pair, train)
                if bundle['layer'] != layer:
                    raise ValueError('Saved model layer mismatch')
                model_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                model = bundle['pipeline']
                for test in ('text_chat', 'text_tools'):
                    indices = [i for i,r in enumerate(rows) if r['pair_id'] == pair and r['condition'] == test]
                    scores = model.decision_function(acts[indices, layer])
                    probabilities = model.predict_proba(acts[indices, layer])[:, 1]
                    info = dict(held_out_pair=pair, train_condition=train, test_condition=test, layer=layer)
                    metrics.append(dict(**info, **score_summary(labels[indices], scores)))
                    for i, score, probability in zip(indices, scores, probabilities):
                        predictions.append(dict(**info, run_key=rows[i]['run_key'],
                            task_id=rows[i]['task_id'], true_label=int(labels[i]),
                            decision_score=float(score), probability_harmful=float(probability)))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    (output/'provenance.json').write_text(json.dumps(dict(results_sha256=digest,
        model_sha256=model_hashes, method='Reuse existing training-only scaler and logistic fit; threshold zero; no tuning',
        auc='Unsquashed decision scores; average within-pair AUC, never pool incomparable fold margins',
        interpretation='Exploratory label decoding, not causal danger recognition or refusal'), indent=2))
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.artifact_root, args.probes, args.output)


if __name__ == '__main__':
    main()
