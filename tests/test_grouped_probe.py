import numpy as np
import pytest
import torch

from src.analysis.grouped_probe import (fit_layer, fit_lexical, grouped_splits, prompt_text,
                                       original_coordinate_weights, validate_rows)
from src.analysis.refusal_plot import load_activations


def design():
    return [dict(run_key=f"{pair}-{condition}-{label}-{variant}", pair_id=f"pair-{pair}",
                 condition=condition, harm_label=label, variant_id=str(variant), activation_path="act.pt")
            for pair in range(5) for condition in ("text_chat", "text_tools")
            for label in ("harmful", "harmless") for variant in range(3)]


def test_whole_pair_held_out_for_every_transfer_scheme():
    rows = design()
    splits = list(grouped_splits(rows))
    assert len(splits) == 15
    assert splits == list(grouped_splits(rows))
    assert [split["held_out_pair"] for split in splits] == [f"pair-{pair}" for pair in range(5) for _ in range(3)]
    for split in splits:
        train, test = split["train_indices"], split["test_indices"]
        assert len(train) == 24 and len(test) == 6
        assert {rows[i]["pair_id"] for i in train}.isdisjoint({rows[i]["pair_id"] for i in test})
        assert {rows[i]["pair_id"] for i in test} == {split["held_out_pair"]}
        assert {rows[i]["condition"] for i in train} == {split["train_condition"]}
        assert {rows[i]["condition"] for i in test} == {split["test_condition"]}
        assert set(split["train_run_keys"]).isdisjoint(split["test_run_keys"])


@pytest.mark.parametrize("change", ["error", "truncated", "duplicate", "wrongshape", "missingpair"])
def test_rejects_incomplete_or_unusable_design(change):
    rows = design()
    if change == "error":
        rows[0]["error"] = "capture failed"
    elif change == "truncated":
        rows[0]["truncated"] = True
    elif change == "duplicate":
        rows[0]["run_key"] = rows[1]["run_key"]
    elif change == "wrongshape":
        rows[0]["activation_shape"] = [35, 4096]
    else:
        rows[0]["pair_id"] = "pair-1"
    with pytest.raises(ValueError):
        validate_rows(rows)


def test_scaling_is_training_only_and_original_coordinate_scores_agree():
    pytest.importorskip("sklearn")
    features = np.array([[-3., 10.], [-1., 10.], [1., 10.], [3., 10.]])
    labels = np.array([0, 0, 1, 1])
    pipeline = fit_layer(features, labels)
    heldout = np.array([[-1000., 9000.], [1000., -9000.]])
    scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, [0., 10.])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(5.), 1.])
    before = scaler.mean_.copy()
    np.testing.assert_array_equal(pipeline.predict(heldout), [0, 1])
    np.testing.assert_array_equal(scaler.mean_, before)
    weights = original_coordinate_weights(pipeline)
    intercept = pipeline.named_steps["probe"].intercept_[0] - np.dot(weights, scaler.mean_)
    np.testing.assert_allclose(pipeline.decision_function(heldout), heldout @ weights + intercept)


def test_nonfinite_training_features_rejected():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="finite"):
        fit_layer([[np.nan], [1.]], [0, 1])


def test_lexical_vocabulary_is_training_only_and_saved_prompt_is_source():
    pytest.importorskip("sklearn")
    model = fit_lexical(["move red object", "move red thing", "leave blue object", "leave blue thing"], [1, 1, 0, 0])
    model.predict(["heldoutonlyword red object"])
    assert "heldoutonlyword" not in model.named_steps["tfidf"].vocabulary_
    assert "red object" in model.named_steps["tfidf"].vocabulary_
    row = {"text": "generated response must not be used", "messages": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": [{"type": "text", "text": "User prompt"},
                                       {"type": "image_url", "image_url": {"url": "secretimage"}}]}]}
    assert prompt_text(row) == "System prompt\nUser prompt"
    assert prompt_text({"text": "response only"}) is None


@pytest.mark.parametrize("bad", ["shape", "nonfinite"])
def test_activation_files_reject_wrong_shape_and_nonfinite_values(tmp_path, bad):
    tensor = torch.zeros(36, 4096) if bad == "nonfinite" else torch.zeros(35, 4096)
    if bad == "nonfinite":
        tensor[3, 9] = float("nan")
    torch.save(tensor, tmp_path / "activation.pt")
    with pytest.raises(ValueError):
        load_activations([{"activation_path": "/workspace/results/activation.pt"}],
                         tmp_path / "results.jsonl", tmp_path)
