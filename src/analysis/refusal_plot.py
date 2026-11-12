"""Exploratory harmful-minus-harmless activation contrast, with no layer selection.

Run from the repository root with ``python -m src.analysis.refusal_plot --help``.
This contrast is not a validated causal refusal direction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import torch

LABELS = ("harmful", "harmless")
EXPECTED_SHAPE = (36, 4096)
DISCLAIMER = "Exploratory harm contrast; not a validated causal refusal direction"


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    for row in rows:
        if row.get("harm_label") not in LABELS:
            raise ValueError(f"Invalid harm_label: {row.get('harm_label')!r}")
    return rows


def resolve_activation_path(value, manifest, artifact_root=None):
    """Only remap explicit /workspace/results paths, rejecting traversal."""
    if not isinstance(value, str) or not value:
        raise ValueError("Every row must have an activation_path")
    normalized = value.replace("\\", "/")
    remote = PurePosixPath(normalized)
    if ".." in remote.parts:
        raise ValueError("Parent traversal is not allowed in activation_path")
    if artifact_root is not None and remote.parts[:3] == ("/", "workspace", "results"):
        root = Path(artifact_root).resolve()
        result = root.joinpath(*remote.parts[3:]).resolve()
        if not result.is_relative_to(root):
            raise ValueError("Activation path escapes artifact root")
        return result
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).parent / path


def load_activations(rows, manifest, artifact_root=None, expected_shape=EXPECTED_SHAPE):
    tensors = []
    for row in rows:
        path = resolve_activation_path(row.get("activation_path"), manifest, artifact_root)
        if row.get('activation_sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != row['activation_sha256']:
            raise ValueError(f'Activation hash mismatch: {path}')
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(f"Expected activation tensor {expected_shape} at {path}")
        tensor = tensor.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite activation at {path}")
        tensors.append(tensor)
    return torch.stack(tensors)


def compute_directions(activations, labels):
    """Return a unit harmful-minus-harmless mean contrast for every layer."""
    activations = torch.as_tensor(activations, dtype=torch.float64, device="cpu")
    if activations.ndim != 3 or len(labels) != len(activations):
        raise ValueError("Expected [samples, layers, hidden] and one label per sample")
    if not torch.isfinite(activations).all() or any(label not in LABELS for label in labels):
        raise ValueError("Activations must be finite and labels harmful/harmless")
    if any(label not in labels for label in LABELS):
        raise ValueError("Calibration requires harmful and harmless samples")
    means = {label: activations[[i for i, value in enumerate(labels) if value == label]].mean(0) for label in LABELS}
    contrast = means["harmful"] - means["harmless"]
    norms = torch.linalg.vector_norm(contrast, dim=-1)
    bad = torch.nonzero(norms <= 1e-12).flatten().tolist()
    if bad:
        raise ValueError(f"Zero or near-zero contrast norm at zero-based layers {bad}")
    return contrast / norms[:, None]


def score_activations(activations, directions):
    activations = torch.as_tensor(activations, dtype=torch.float64, device="cpu")
    directions = torch.as_tensor(directions, dtype=torch.float64, device="cpu")
    if activations.ndim != 3 or activations.shape[1:] != directions.shape:
        raise ValueError("Activation/direction shape mismatch")
    if not torch.isfinite(activations).all() or not torch.isfinite(directions).all():
        raise ValueError("Scores require finite inputs")
    return torch.einsum("nlh,lh->nl", activations, directions)


def analyze(calibration, results, output_dir, artifact_root=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Use a fresh analysis directory to preserve prior evidence')
    calibration_rows, rows = read_rows(calibration), read_rows(results)
    if any(row.get('error') or row.get('truncated') for row in rows):
        raise ValueError('Evaluation contains errors or pending truncation; resolve before analysis')
    if any(not isinstance(row.get("condition"), str) or not row["condition"] for row in rows):
        raise ValueError("Each burner row requires a nonempty condition")
    directions = compute_directions(load_activations(calibration_rows, calibration, artifact_root),
                                    [row["harm_label"] for row in calibration_rows])
    scores = score_activations(load_activations(rows, results, artifact_root), directions)
    groups = defaultdict(list)
    records = []
    for index, (row, values) in enumerate(zip(rows, scores.tolist())):
        groups[(row["condition"], row["harm_label"])].append(values)
        records.append({"row_index": index, "task_id": row.get("task_id"),
                        "repeat": row.get("repeat"), "condition": row["condition"],
                        "harm_label": row["harm_label"], "scores": values})
    summaries = [{"condition": condition, "harm_label": label, "n": len(values),
                  "mean_scores": torch.tensor(values, dtype=torch.float64).mean(0).tolist()}
                 for (condition, label), values in sorted(groups.items())]
    payload = {"interpretation": DISCLAIMER, "score_definition": "activation dot unit(harmful mean - harmless mean)",
               "layer_indexing": "zero-based; all layers shown; no layer selection",
               "calibration_counts": dict(Counter(row["harm_label"] for row in calibration_rows)),
               "calibration_truncated_count": sum(bool(row.get('truncated')) for row in calibration_rows),
               "samples": records, "groups": summaries}
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'calibration_sha256': hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                  'results_sha256': hashlib.sha256(Path(results).read_bytes()).hexdigest(),
                  'activation_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for records, manifest in ((calibration_rows, calibration), (rows, results))
                     for row in records
                     for path in [resolve_activation_path(row['activation_path'], manifest, artifact_root)]},
                  'score_definition': payload['score_definition'], 'interpretation': DISCLAIMER}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    torch.save(directions, output / "direction.pt")
    (output / "scores.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "task_id", "repeat", "condition", "harm_label", "layer", "score"])
        for record in records:
            for layer, score in enumerate(record["scores"]):
                writer.writerow([record[key] for key in ("row_index", "task_id", "repeat", "condition", "harm_label")] + [layer, score])
    plot_scores(payload, output)
    return payload


def plot_scores(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted({group["condition"] for group in payload["groups"]})
    colors = dict(zip(conditions, plt.get_cmap("tab10").colors * (1 + len(conditions) // 10)))
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
        for axis, label, title in zip(axes, LABELS, ("Harmful burner tasks", "Harmless burner tasks (safe)")):
            matched = [group for group in payload["groups"] if group["harm_label"] == label]
            for group in matched:
                axis.plot(range(len(group["mean_scores"])), group["mean_scores"],
                          label=f"{group['condition']} (n={group['n']})", color=colors[group["condition"]], linewidth=1.8)
            axis.axhline(0, color="0.55", linewidth=0.7, linestyle="--")
            axis.set(title=title, xlabel="Transformer layer (zero-based)")
            axis.grid(alpha=0.2)
            if matched:
                axis.legend(frameon=False, fontsize=8)
            else:
                axis.text(0.5, 0.5, "No samples", transform=axis.transAxes, ha="center")
        axes[0].set_ylabel("Mean projection onto unit harm contrast")
        fig.suptitle("Burner activation projections across all layers")
        counts = payload["calibration_counts"]
        fig.text(0.5, 0.025, f"{DISCLAIMER}\nCalibration: harmful n={counts.get('harmful', 0)}, harmless n={counts.get('harmless', 0)}; no layer selection.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 0.94))
        for suffix in ("png", "svg"):
            fig.savefig(Path(output) / f"refusal_plot.{suffix}", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, help="Local replacement for /workspace/results/ only")
    args = parser.parse_args()
    analyze(args.calibration, args.results, args.output_dir, args.artifact_root)


if __name__ == "__main__":
    main()
