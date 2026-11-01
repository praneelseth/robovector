import pytest
import torch
from torch import nn

from robot_refusal.activations import CaptureError, ResidualHooks
from robot_refusal.records import CaptureConfig, InterventionConfig


class AddBlock(nn.Module):
    def __init__(self, value=1, tuple_output=False):
        super().__init__()
        self.value = value
        self.tuple_output = tuple_output

    def forward(self, hidden):
        result = hidden + self.value
        return (result, "preserved") if self.tuple_output else result


def make_hooks(block, capture=None, intervention=None, ids=None, mask=None):
    return ResidualHooks({0: ("decoder.layers.0", block)}, 3, ids or [10, 11, 0, 0],
        mask or [1, 1, 0, 0], capture or CaptureConfig(enabled=True), intervention)


def test_prompt_index_and_cached_token_alignment():
    block = AddBlock()
    prompt = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    capture = CaptureConfig(enabled=True, generated_token_indices=[0, 1, 3])
    hooks = make_hooks(block, capture)
    with hooks:
        block(prompt)
        block(torch.full((1, 1, 3), 9.0))
    hooks.finalize([50, 51])
    assert len(block._forward_hooks) == 0
    assert torch.equal(hooks.tensors["layer_0.prompt_1.residual"], prompt[0, 1] + 1)
    assert torch.equal(hooks.tensors["layer_0.generation_0.residual"], torch.full((3,), 10.0))
    assert hooks.metadata[0]["token_id"] == 11
    assert hooks.metadata[0]["sequence_position"] == 1
    assert hooks.metadata[1]["token_id"] == 50
    assert hooks.metadata[1]["sequence_position"] == 4
    assert hooks.unavailable == [
        {"layer": 0, "generated_token_index": 1, "reason": "sampled_but_not_forwarded"},
        {"layer": 0, "generated_token_index": 3, "reason": "not_generated"},
    ]
    assert all(t.device.type == "cpu" and not t.requires_grad for t in hooks.tensors.values())


def test_one_token_prefill_is_not_cached_generation():
    block = AddBlock()
    hooks = make_hooks(block, ids=[8], mask=[1])
    with hooks:
        block(torch.zeros(1, 1, 3))
    hooks.finalize([9])
    assert hooks.metadata[0]["stage"] == "prompt"
    assert hooks.metadata[0]["token_id"] == 8


@pytest.mark.parametrize("capture", [CaptureConfig(), CaptureConfig(enabled=True)])
def test_capture_output_identity(capture):
    block = AddBlock()
    x = torch.randn(1, 4, 3)
    baseline = block(x)
    with make_hooks(block, capture) as hooks:
        observed = block(x)
    hooks.finalize([5])
    assert torch.equal(observed, baseline)
    assert bool(hooks.tensors) == capture.enabled
    assert not block._forward_hooks


@pytest.mark.parametrize("strength", [0.0, 2.0])
@pytest.mark.parametrize("schedule", ["prompt_boundary", "prompt_boundary_and_generation"])
def test_intervention_values_schedule_and_identity(tmp_path, strength, schedule):
    block = AddBlock(tuple_output=True)
    path = tmp_path / "direction.pt"
    direction = torch.tensor([1.0, 2.0, 3.0])
    torch.save(direction, path)
    intervention = InterventionConfig(layer=0, direction_path=str(path), strength=strength, schedule=schedule)
    x = torch.zeros(1, 4, 3)
    baseline = block(x)[0]
    hooks = make_hooks(block, CaptureConfig(enabled=True, generated_token_indices=[0]), intervention)
    with hooks:
        actual, other = block(x)
        generated, _ = block(torch.zeros(1, 1, 3))
    hooks.finalize([5, 6])
    assert other == "preserved"
    assert torch.equal(actual[0, 1], baseline[0, 1] + strength * direction)
    assert torch.equal(actual[0, [0, 2, 3]], baseline[0, [0, 2, 3]])
    assert torch.equal(x, torch.zeros_like(x))
    expected_generated = torch.ones(3) + (strength * direction if schedule.endswith("and_generation") else 0)
    assert torch.equal(generated[0, 0], expected_generated)
    if strength == 0:
        assert torch.equal(actual, baseline)
    assert torch.equal(hooks.tensors["layer_0.prompt_1.pre_intervention"], torch.ones(3))
    assert torch.equal(hooks.tensors["layer_0.prompt_1.post_intervention"], torch.ones(3) + strength * direction)
    assert hooks.direction_sha256
    assert not block._forward_hooks


def test_intervention_captures_pre_post_even_if_capture_disabled(tmp_path):
    path = tmp_path / "direction.pt"
    torch.save(torch.ones(3), path)
    block = AddBlock()
    with make_hooks(block, CaptureConfig(), InterventionConfig(layer=0, direction_path=str(path))) as hooks:
        block(torch.zeros(1, 4, 3))
    assert {row["phase"] for row in hooks.metadata} == {"pre_intervention", "post_intervention"}


def test_cleanup_on_shape_error_and_external_exception():
    block = AddBlock()
    with pytest.raises(CaptureError):
        with make_hooks(block):
            block(torch.zeros(1, 4, 2))
    assert not block._forward_hooks
    with pytest.raises(RuntimeError, match="generation failed"):
        with make_hooks(block):
            block(torch.zeros(1, 4, 3))
            raise RuntimeError("generation failed")
    assert not block._forward_hooks


def test_missing_hook_and_missing_forwarded_capture_fail():
    block = AddBlock()
    with make_hooks(block) as hooks:
        pass
    with pytest.raises(CaptureError, match="never called"):
        hooks.finalize([1])
    with make_hooks(block, CaptureConfig(enabled=True, generated_token_indices=[0])) as hooks:
        block(torch.zeros(1, 4, 3))
    with pytest.raises(CaptureError, match="Missing forwarded"):
        hooks.finalize([1, 2])


def test_invalid_layer_and_uncached_generation_rejected():
    block = AddBlock()
    with pytest.raises(CaptureError, match="Unknown decoder"):
        make_hooks(block, CaptureConfig(enabled=True, layers=[7]))
    with pytest.raises(CaptureError, match="sequence length 1"):
        with make_hooks(block):
            block(torch.zeros(1, 4, 3))
            block(torch.zeros(1, 5, 3))
    assert not block._forward_hooks


@pytest.mark.parametrize("direction", [torch.ones(4), torch.tensor([float("nan"), 1.0, 1.0]),
                                       torch.tensor([1, 2, 3]), {"tensor": torch.ones(3)}])
def test_direction_validation(tmp_path, direction):
    path = tmp_path / "invalid.pt"
    torch.save(direction, path)
    with pytest.raises(CaptureError):
        make_hooks(AddBlock(), intervention=InterventionConfig(layer=0, direction_path=str(path)))


def test_all_layers_capture_only_selected_positions():
    blocks = nn.Sequential(AddBlock(1), AddBlock(2))
    modules = {i: (f"layers.{i}", layer) for i, layer in enumerate(blocks)}
    with ResidualHooks(modules, 3, [1, 2], [1, 1], CaptureConfig(enabled=True)) as hooks:
        blocks(torch.zeros(1, 2, 3))
    hooks.finalize([7])
    assert len(hooks.tensors) == 2
    assert torch.equal(hooks.tensors["layer_0.prompt_1.residual"], torch.ones(3))
    assert torch.equal(hooks.tensors["layer_1.prompt_1.residual"], torch.full((3,), 3.0))
    assert all(not block._forward_hooks for block in blocks)
