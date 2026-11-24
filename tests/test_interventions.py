import pytest
import torch
from torch import nn

from src.analysis.interventions import AllPositionIntervention, seeded_random_direction


class IdentityBlock(nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output
        self.marker = object()

    def forward(self, hidden):
        return (hidden, self.marker, 'cache') if self.tuple_output else hidden


def mapping(*blocks):
    return {i: (f'layers.{i}', block) for i, block in enumerate(blocks)}


def test_addition_changes_every_position_at_one_layer_including_cached_decode():
    first, second = IdentityBlock(), IdentityBlock()
    direction = torch.tensor([1., 2., 3.])
    prompt = torch.randn(2, 4, 3)
    saved = prompt.clone()
    with AllPositionIntervention(mapping(first, second), direction, layer=1, strength=2):
        assert torch.equal(first(prompt), prompt)
        assert torch.equal(second(prompt), prompt + 2*direction)
        assert torch.equal(second(torch.zeros(1, 1, 3)), (2*direction).reshape(1,1,3))
    assert torch.equal(prompt, saved)
    assert not first._forward_hooks and not second._forward_hooks


def test_ablation_removes_projection_at_all_layers_positions_and_is_idempotent():
    first, second = IdentityBlock(), IdentityBlock()
    direction = torch.tensor([2., 3., 4.], dtype=torch.float64)
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    unit = direction / direction.norm()
    expected = x - (x*unit).sum(-1,keepdim=True)*unit
    with AllPositionIntervention(mapping(first, second), direction, mode='ablation'):
        one = first(x)
        two = second(one)
    torch.testing.assert_close(one, expected)
    torch.testing.assert_close(two, expected)
    torch.testing.assert_close((two*unit).sum(-1), torch.zeros(2,5,dtype=torch.float64), atol=1e-14, rtol=0)


def test_single_layer_fractional_ablation_preserves_untargeted_layer():
    first, second = IdentityBlock(), IdentityBlock()
    x = torch.tensor([[[3., 4.]]])
    with AllPositionIntervention(mapping(first,second), torch.tensor([2.,0.]), mode='ablation', layer=0, strength=.5):
        assert torch.equal(first(x), torch.tensor([[[1.5,4.]]]))
        assert second(x) is x


@pytest.mark.parametrize('mode', ['addition','ablation'])
def test_zero_strength_preserves_exact_tuple_and_tensor_identity(mode):
    block = IdentityBlock(tuple_output=True)
    x = torch.randn(1,3,2)
    outputs = []
    # Earlier observer stores the exact tuple the intervention receives.
    handle = block.register_forward_hook(lambda _m,_i,o: outputs.append(o))
    with AllPositionIntervention(mapping(block), torch.ones(2), mode=mode, layer=0, strength=0):
        result = block(x)
    handle.remove()
    assert result is outputs[0]
    assert result[0] is x


def test_tuple_trailing_fields_and_dtype_preserved():
    block = IdentityBlock(tuple_output=True)
    x = torch.randn(1,4,3,dtype=torch.bfloat16)
    for mode in ['addition','ablation']:
        with AllPositionIntervention(mapping(block), torch.ones(3,dtype=torch.float64), mode=mode, layer=0):
            out = block(x)
        assert out[0].dtype == x.dtype and out[0].device == x.device
        assert out[1] is block.marker and out[2] == 'cache'


def test_seeded_random_norm_reproducibility_and_no_global_rng_changes():
    reference = torch.tensor([3.,4.,0.,0.],dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    first = seeded_random_direction(reference,42)
    assert torch.equal(first,seeded_random_direction(reference,42))
    assert not torch.equal(first,seeded_random_direction(reference,43))
    torch.testing.assert_close(first.norm(), reference.norm())
    assert torch.equal(before,torch.random.get_rng_state())


def test_cleanup_on_external_and_shape_errors():
    block = IdentityBlock()
    for external in [True,False]:
        with pytest.raises((ValueError,RuntimeError)):
            with AllPositionIntervention(mapping(block),torch.ones(3),layer=0):
                if external:
                    raise RuntimeError('generation failure')
                block(torch.zeros(1,2,4))
        assert not block._forward_hooks


def test_later_capture_observer_sees_post_intervention_output():
    block = IdentityBlock()
    observed = []
    with AllPositionIntervention(mapping(block),torch.tensor([1.,2.]),layer=0):
        handle = block.register_forward_hook(lambda _m,_i,o: observed.append(o.detach().clone()))
        try:
            result = block(torch.zeros(1,2,2))
        finally:
            handle.remove()
    assert torch.equal(observed[0], result)
    assert torch.equal(result, torch.tensor([[[1.,2.],[1.,2.]]]))


@pytest.mark.parametrize('mode', ['addition','ablation'])
def test_cached_direction_reused_across_tokens_and_released_on_close(mode):
    first, second = IdentityBlock(), IdentityBlock()
    vector = torch.tensor([2.,3.,4.],dtype=torch.float64)
    context = AllPositionIntervention(mapping(first,second),vector,mode=mode,
                                     layer=0 if mode == 'addition' else None)
    with context:
        x = torch.randn(1,4,3)
        first(x)
        cached = next(iter(context._device_cache.values()))
        # Prefill, cached decoding, and another layer share one converted vector.
        first(torch.randn(1,1,3))
        if mode == 'ablation':
            second(x)
        assert len(context._device_cache) == 1
        assert next(iter(context._device_cache.values())) is cached
        x64 = x.double()
        observed = first(x64)
        if mode == 'addition':
            expected = x64 + vector
        else:
            unit = vector / vector.norm()
            expected = x64 - (x64*unit).sum(-1,keepdim=True)*unit
        torch.testing.assert_close(observed,expected)
        assert len(context._device_cache) == 2
    assert context._device_cache == {}
    assert all(not block._forward_hooks for block in [first,second])


def test_zero_sham_never_populates_device_cache_and_error_cleanup_releases_it():
    block = IdentityBlock()
    sham = AllPositionIntervention(mapping(block),torch.ones(3),layer=0,strength=0)
    with sham:
        block(torch.zeros(1,2,3))
        assert sham._device_cache == {}
    active = AllPositionIntervention(mapping(block),torch.ones(3),layer=0)
    with pytest.raises(RuntimeError):
        with active:
            block(torch.zeros(1,2,3))
            assert active._device_cache
            raise RuntimeError('generation failure')
    assert active._device_cache == {}


@pytest.mark.parametrize('bad_id',[-1,True,'0'])
def test_invalid_module_ids_fail(bad_id):
    with pytest.raises(ValueError):
        AllPositionIntervention({bad_id:('block',IdentityBlock())},torch.ones(3),mode='ablation')


@pytest.mark.parametrize('direction',[torch.zeros(3), torch.ones(2,2), torch.ones(3,dtype=torch.long),
                                      torch.tensor([float('nan'),1.])])
def test_invalid_directions_fail(direction):
    with pytest.raises(ValueError):
        AllPositionIntervention(mapping(IdentityBlock()),direction,layer=0)


@pytest.mark.parametrize('kwargs',[dict(mode='other',layer=0),dict(layer=None),dict(layer=2),
                                  dict(layer=True),dict(layer=0,strength=float('inf'))])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        AllPositionIntervention(mapping(IdentityBlock()),torch.ones(3),**kwargs)
