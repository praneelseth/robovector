"""Opt-in checkpoint tests. They never fetch weights or synthesize scene photos."""

import os

import pytest
import torch

from robot_refusal.agent import generation_settings
from robot_refusal.prompts import build_messages
from robot_refusal.records import AgentTrial, CaptureConfig, InterventionConfig, ModelConfig

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def real_adapter():
    model_dir = os.environ.get("ROBOT_REFUSAL_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set ROBOT_REFUSAL_MODEL_DIR to opt in to local checkpoint integration")
    from robot_refusal.models.qwen_vl import QwenVLAdapter
    adapter = QwenVLAdapter()
    try:
        adapter.load(ModelConfig(model_dir=model_dir, device=os.environ.get("ROBOT_REFUSAL_DEVICE", "auto")))
        yield adapter
    finally:
        adapter.close()


@pytest.mark.parametrize("condition", ["C1", "C2", "C3", "C4"])
def test_real_condition_and_capture_identity(real_adapter, trial_data, tmp_path, condition):
    image = os.environ.get("ROBOT_REFUSAL_TEST_IMAGE")
    if condition in ("C3", "C4") and not image:
        pytest.skip("Set ROBOT_REFUSAL_TEST_IMAGE to a local photograph for C3/C4 integration")
    # This smoke test checks inference plumbing, not recognition or behavioral scoring.
    trial = AgentTrial.model_validate({**trial_data, "condition": condition,
        "instruction": "Describe the supplied scene briefly.", "max_new_tokens_per_turn": 16,
        "image_paths": [image] if condition in ("C3", "C4") else []})
    trial.check_input_files()
    messages, _ = build_messages(trial)
    prepared = real_adapter.prepare(messages, generation_settings(trial))
    assert bool(prepared.metadata["image_preprocessing"]["images"]) == trial.uses_images
    baseline = real_adapter.generate(prepared, CaptureConfig())
    capture = CaptureConfig(enabled=True, generated_token_indices=[0, 1, 15])
    observed = real_adapter.generate(prepared, capture)
    assert observed.capture_error is None
    assert baseline.generated_ids == observed.generated_ids
    prompt_rows = [row for row in observed.activation_metadata if row["stage"] == "prompt"]
    assert len(prompt_rows) == len(real_adapter.modules)
    assert {row["sequence_position"] for row in prompt_rows} == {prepared.metadata["prompt_boundary"]["position"]}
    assert {row["token_id"] for row in prompt_rows} == {prepared.metadata["prompt_boundary"]["token_id"]}
    direction = tmp_path / "zero-check-direction.pt"
    torch.save(torch.ones(real_adapter.hidden_size), direction)
    zero = real_adapter.generate(prepared, capture, InterventionConfig(layer=0, direction_path=str(direction),
        strength=0.0, schedule="prompt_boundary_and_generation"))
    assert zero.capture_error is None
    assert zero.generated_ids == baseline.generated_ids
    for _, module in real_adapter.modules.values():
        assert not module._forward_hooks
    assert {row["phase"] for row in zero.activation_metadata if row["layer"] == 0} == {"pre_intervention", "post_intervention"}
    assert observed.resolved_configuration["model_commit"] == real_adapter.snapshot["model_commit"]
