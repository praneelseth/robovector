"""Exploratory pair-held-out harm-label probes; no layer or parameter selection.

Run ``python -m src.analysis.grouped_probe --help``. This is a harm-label
diagnostic on five task pairs, not evidence of a causal refusal representation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .refusal_plot import EXPECTED_SHAPE, load_activations, read_rows

CONDITIONS = ("text_chat", "text_tools")
SCHEMES = (("text_chat", "text_chat"), ("text_chat", "text_tools"),
           ("text_tools", "text_tools"))
PROBE_CONFIG = dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
INTERPRETATION = (
    "Exploratory five-pair harm-label decoding. All layers reported; no layer or "
    "hyperparameter selection, significance test, or validated causal refusal claim. "
    "Paraphrases are not independent task pairs. Failure does not establish H1."
)


def validate_rows(rows):
    """Require the complete 5 x 2 labels x 3 variants x 2 conditions design."""
    if len(rows) != 60:
        raise ValueError("Expected exactly 60 rows: five complete pairs, three variants, two labels/conditions")
    seen = set()
    counts = Counter()
    for row in rows:
        if row.get("error") or row.get("truncated") or row.get("outcome") == "error":
            raise ValueError("Error or truncated rows cannot enter probes; complete the source run first")
        for key in ("run_key", "pair_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Missing {key}")
        if row["run_key"] in seen:
            raise ValueError("Duplicate run_key")
        seen.add(row["run_key"])
        if row.get("condition") not in CONDITIONS or row.get("harm_label") not in ("harmful", "harmless"):
            raise ValueError("Expected text_chat/text_tools and harmful/harmless labels")
        if not row.get("activation_path"):
            raise ValueError("Missing activation_path")
        if row.get("activation_shape") is not None and tuple(row["activation_shape"]) != EXPECTED_SHAPE:
            raise ValueError(f"Expected activation_shape {EXPECTED_SHAPE}")
        counts[(row["pair_id"], row["condition"], row["harm_label"])] += 1
    pairs = sorted({row["pair_id"] for row in rows})
    if len(pairs) != 5 or any(counts[(pair, condition, label)] != 3
                              for pair in pairs for condition in CONDITIONS
                              for label in ("harmful", "harmless")):
        raise ValueError("Each of five pairs must have three examples per condition and harm label")
    return pairs


def grouped_splits(rows):
    """Every test example belongs to the one pair absent from all training rows."""
    pairs = validate_rows(rows)
    for held_pair in pairs:
        for train_condition, test_condition in SCHEMES:
            train = [i for i, row in enumerate(rows)
                     if row["pair_id"] != held_pair and row["condition"] == train_condition]
            test = [i for i, row in enumerate(rows)
                    if row["pair_id"] == held_pair and row["condition"] == test_condition]
            yield dict(held_out_pair=held_pair, train_condition=train_condition,
                       test_condition=test_condition, train_indices=train, test_indices=test,
                       train_run_keys=[rows[i]["run_key"] for i in train],
                       test_run_keys=[rows[i]["run_key"] for i in test],
                       train_pairs=[pair for pair in pairs if pair != held_pair])


def fit_layer(train_features, train_labels):
    """Fit the scaler and fixed probe exclusively on the training partition."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_features = np.asarray(train_features, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if train_features.ndim != 2 or not np.isfinite(train_features).all():
        raise ValueError("Training features must be a finite matrix")
    if len(train_labels) != len(train_features) or set(train_labels.tolist()) != {0, 1}:
        raise ValueError("Training requires both harm labels")
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("probe", LogisticRegression(**PROBE_CONFIG))])
    # A nonconverged fixed fit is reported as a failure, never silently used or
    # repaired by tuning parameters after seeing the held-out pair.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(train_features, train_labels)
    return pipeline


def original_coordinate_weights(pipeline):
    """Convert standardized-feature coefficients back to activation coordinates."""
    return pipeline.named_steps["probe"].coef_[0] / pipeline.named_steps["scaler"].scale_


def cosine_similarity(left, right):
    left, right = np.asarray(left), np.asarray(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def prompt_text(row):
    """Extract saved prompt text parts; never substitute generated responses."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part["text"] for part in content
                         if isinstance(part, dict) and part.get("type") == "text"
                         and isinstance(part.get("text"), str))
    return "\n".join(parts) or None


def fit_lexical(texts, labels):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                         ("probe", LogisticRegression(**PROBE_CONFIG))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(texts, labels)
    return pipeline


def lexical_baseline(rows, splits, output):
    """Use identical pair folds and fit vocabulary/IDF only on training text."""
    import joblib
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    texts = [prompt_text(row) for row in rows]
    if any(text is None for text in texts):
        return dict(status="unavailable", reason="Saved row.messages prompt text is missing; responses were not used")
    labels = np.array([int(row["harm_label"] == "harmful") for row in rows])
    metrics, predictions, models = [], [], {}
    for split in splits:
        train, test = split["train_indices"], split["test_indices"]
        key = (split["held_out_pair"], split["train_condition"])
        if key not in models:
            model = fit_lexical([texts[i] for i in train], labels[train])
            filename = f"lexical-fold-{len(models)}_{split['train_condition']}.joblib"
            joblib.dump(dict(pipeline=model, split=split), output / "models" / filename)
            models[key] = (model, filename)
        model, filename = models[key]
        predicted = model.predict([texts[i] for i in test])
        probabilities = model.predict_proba([texts[i] for i in test])[:, 1]
        metric = {key: split[key] for key in ("held_out_pair", "train_condition", "test_condition")}
        metric.update(n_train=len(train), n_test=len(test),
                      accuracy=float(accuracy_score(labels[test], predicted)),
                      balanced_accuracy=float(balanced_accuracy_score(labels[test], predicted)),
                      roc_auc=float(roc_auc_score(labels[test], probabilities)), model_artifact=f"models/{filename}")
        metrics.append(metric)
        for i, prediction, probability in zip(test, predicted, probabilities):
            record = {key: rows[i].get(key) for key in ("run_key", "pair_id", "variant_id", "condition", "harm_label")}
            record.update(held_out_pair=split["held_out_pair"], train_condition=split["train_condition"],
                          true_label=int(labels[i]), predicted_label=int(prediction), probability_harmful=float(probability))
            predictions.append(record)
    (output / "lexical_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "lexical_predictions.jsonl").write_text("".join(json.dumps(record) + "\n" for record in predictions), encoding="utf-8")
    return dict(status="complete", n_splits=len(metrics), ngram_range=[1, 2], max_features=10000,
                probe_config=PROBE_CONFIG, source="saved row.messages text parts only; prompt system/user/tool text included")


def analyze(results, output_dir, artifact_root=None, reference_direction=None):
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Output directory must be fresh; existing analyses are never overwritten")
    rows = read_rows(results)
    splits = list(grouped_splits(rows))
    activations = load_activations(rows, results, artifact_root).numpy()
    labels = np.array([int(row["harm_label"] == "harmful") for row in rows])
    reference = None
    if reference_direction is not None:
        tensor = torch.load(reference_direction, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != EXPECTED_SHAPE or not torch.isfinite(tensor).all():
            raise ValueError(f"Reference harm contrast must be finite with shape {EXPECTED_SHAPE}")
        reference = tensor.detach().double().cpu().numpy()
        if np.any(np.linalg.norm(reference, axis=1) <= 1e-12):
            raise ValueError("Reference harm contrast has a zero-norm layer")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    metadata = dict(interpretation=INTERPRETATION, probe_config=PROBE_CONFIG,
                    scaler="StandardScaler fit only on fold training examples", label_encoding={"harmless": 0, "harmful": 1},
                    decision_threshold=0.5, layer_indexing="zero-based; every layer; no selection",
                    split="leave-one-whole-pair-out; same held-out pair for each transfer scheme",
                    n_rows=len(rows), n_pairs=5, sklearn_version=sklearn.__version__,
                    results_sha256=hashlib.sha256(Path(results).read_bytes()).hexdigest(),
                    reference_direction_sha256=(hashlib.sha256(Path(reference_direction).read_bytes()).hexdigest()
                                                if reference_direction is not None else None),
                    reference_interpretation="Unit harm contrast; not a validated causal refusal direction",
                    status="running")
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics = []
    # Cache each fold/training-condition/layer fit for transfer without refitting.
    models = {}
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
            for split in splits:
                train, test = split["train_indices"], split["test_indices"]
                fold = sorted({row["pair_id"] for row in rows}).index(split["held_out_pair"])
                for layer in range(activations.shape[1]):
                    key = (fold, split["train_condition"], layer)
                    if key not in models:
                        model = fit_layer(activations[train, layer], labels[train])
                        filename = f"fold-{fold}_{split['train_condition']}_layer-{layer:02d}.joblib"
                        joblib.dump(dict(pipeline=model, split=split, layer=layer,
                                         original_coordinate_weights=original_coordinate_weights(model),
                                         interpretation=INTERPRETATION), output / "models" / filename)
                        models[key] = (model, filename)
                    model, filename = models[key]
                    probabilities = model.predict_proba(activations[test, layer])[:, 1]
                    decisions = model.decision_function(activations[test, layer])
                    predicted = model.predict(activations[test, layer])
                    metric = dict(held_out_pair=split["held_out_pair"], train_condition=split["train_condition"],
                                  test_condition=split["test_condition"], layer=layer,
                                  n_train=len(train), n_test=len(test), n_harmful=int(labels[test].sum()),
                                  n_harmless=int(len(test) - labels[test].sum()),
                                  accuracy=float(accuracy_score(labels[test], predicted)),
                                  balanced_accuracy=float(balanced_accuracy_score(labels[test], predicted)),
                                  roc_auc=float(roc_auc_score(labels[test], probabilities)), model_artifact=f"models/{filename}")
                    if reference is not None:
                        metric["cosine_with_reference_harm_contrast"] = cosine_similarity(original_coordinate_weights(model), reference[layer])
                    metrics.append(metric)
                    for i, prediction, probability, decision in zip(test, predicted, probabilities, decisions):
                        record = {key: rows[i].get(key) for key in ("run_key", "pair_id", "task_id", "variant_id", "paraphrase_id", "repeat", "condition", "harm_label")}
                        record.update(held_out_pair=split["held_out_pair"], train_condition=split["train_condition"],
                                      test_condition=split["test_condition"], layer=layer, true_label=int(labels[i]),
                                      predicted_label=int(prediction), probability_harmful=float(probability),
                                      decision_score=float(decision), model_artifact=f"models/{filename}")
                        predictions.write(json.dumps(record) + "\n")
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        try:
            metadata["lexical_baseline"] = lexical_baseline(rows, splits, output)
        except Exception as exc:
            metadata["lexical_baseline"] = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Local root replacing only /workspace/results/ activation paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-direction", type=Path)
    args = parser.parse_args()
    analyze(args.results, args.output_dir, args.artifact_root, args.reference_direction)


if __name__ == "__main__":
    main()
