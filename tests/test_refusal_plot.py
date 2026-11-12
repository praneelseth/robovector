import json

import pytest
import torch

from src.analysis.refusal_plot import analyze, compute_directions, resolve_activation_path, score_activations


def test_known_layer_directions_and_signed_scores():
    calibration = torch.tensor([[[3., 0.], [0., -4.]], [[1., 0.], [0., -2.]],
                                [[0., 0.], [0., 0.]]])
    directions = compute_directions(calibration, ["harmful", "harmful", "harmless"])
    torch.testing.assert_close(directions, torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64))
    burner = torch.tensor([[[2., 9.], [8., 3.]], [[-5., 1.], [2., -7.]]])
    torch.testing.assert_close(score_activations(burner, directions),
                               torch.tensor([[2., -3.], [-5., 7.]], dtype=torch.float64))


def test_zero_norm_layer_rejected_instead_of_silently_normalized():
    with pytest.raises(ValueError, match="zero-based layers \\[1\\]"):
        compute_directions(torch.tensor([[[1., 0.], [2., 3.]], [[0., 0.], [2., 3.]]]),
                           ["harmful", "harmless"])


def test_missing_calibration_class_rejected():
    with pytest.raises(ValueError, match="requires harmful and harmless"):
        compute_directions(torch.ones(2, 3, 4), ["harmful", "harmful"])


def test_remote_path_remapping_is_explicit_and_contained(tmp_path):
    root = tmp_path / "artifacts"
    assert resolve_activation_path("/workspace/results/run/act.pt", "data.jsonl", root) == root / "run" / "act.pt"
    with pytest.raises(ValueError, match="traversal"):
        resolve_activation_path("/workspace/results/../secret.pt", "data.jsonl", root)
    assert resolve_activation_path("acts/a.pt", tmp_path / "results.jsonl", root) == tmp_path / "acts" / "a.pt"


def test_exports_keep_harm_labels_separate(tmp_path):
    calibration_rows, result_rows = [], []
    for label, value in (("harmful", 2.0), ("harmless", -1.0)):
        tensor = torch.zeros(36, 4096)
        tensor[:, 0] = value
        torch.save(tensor, tmp_path / f"{label}.pt")
        row = {"harm_label": label, "activation_path": f"/workspace/results/{label}.pt"}
        calibration_rows.append(row)
        result_rows.append({**row, "condition": "text_chat"})
    calibration = tmp_path / "calibration.jsonl"
    results = tmp_path / "results.jsonl"
    calibration.write_text("\n".join(map(json.dumps, calibration_rows)), encoding="utf-8")
    results.write_text("\n".join(map(json.dumps, result_rows)), encoding="utf-8")
    output = tmp_path / "analysis"
    payload = analyze(calibration, results, output, tmp_path)
    groups = {group["harm_label"]: group for group in payload["groups"]}
    assert groups["harmful"]["mean_scores"] == [2.0] * 36
    assert groups["harmless"]["mean_scores"] == [-1.0] * 36
    assert all(group["n"] == 1 for group in groups.values())
    for name in ("direction.pt", "scores.json", "scores.csv", "refusal_plot.png", "refusal_plot.svg"):
        assert (output / name).stat().st_size > 0
    assert len((output / "scores.csv").read_text().splitlines()) == 73
    assert (output / 'provenance.json').is_file()
    with pytest.raises(FileExistsError):
        analyze(calibration, results, output, tmp_path)


def test_evaluation_failures_are_not_silently_scored(tmp_path):
    p = tmp_path / 'rows.jsonl'
    p.write_text(json.dumps({'harm_label':'harmful', 'truncated':True, 'condition':'vision'}))
    with pytest.raises(ValueError, match='pending truncation'):
        analyze(p, p, tmp_path / 'out')
