from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deploy.runpod.position_capture import (PositionCapture, compare_capture,
                                           independent_forward_check, resolve_positions)


def prepared(text='go X'):
    ids = list(range(10, 10 + len(text)))
    return SimpleNamespace(inputs=dict(input_ids=torch.tensor([ids]),
                                       attention_mask=torch.ones(1, len(ids), dtype=torch.long)),
        metadata=dict(prompt_token_ids=ids, attention_mask=[1]*len(ids), expanded_prompt=text,
                      token_offsets_in_processor_expanded_text=[[i, i+1] for i in range(len(ids))]))


class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class Scale(nn.Module):
    def forward(self, x):
        return x * 10


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30, 3)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([Add(1), Add(2)])
        self.model.language_model.norm = Scale()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        x = self.embedding(input_ids)
        states = [x]
        x = self.model.language_model.layers[0](x)
        states.append(x)
        x = self.model.language_model.layers[1](x)
        states.append(self.model.language_model.norm(x))
        return SimpleNamespace(hidden_states=states)


def owner():
    model = ToyModel()
    return SimpleNamespace(model=model, hidden_size=3,
        modules={i: (f'layers.{i}', block) for i,block in enumerate(model.model.language_model.layers)})


def test_resolves_full_prompt_offsets_and_rejects_ambiguous_instruction():
    assert resolve_positions(prepared(), 'go') == dict(instruction_end=1, prompt_end=3)
    with pytest.raises(ValueError, match='exactly once'):
        resolve_positions(prepared('go go'), 'go')
    with pytest.raises(ValueError, match='exactly once'):
        resolve_positions(prepared(), 'absent')


def test_rejects_missing_cross_span_and_padded_endpoints():
    p = prepared()
    p.metadata['token_offsets_in_processor_expanded_text'][1] = [0,0]
    with pytest.raises(ValueError, match='missing or ambiguous'):
        resolve_positions(p, 'go')
    p = prepared()
    p.metadata['token_offsets_in_processor_expanded_text'][1] = [1,3]
    with pytest.raises(ValueError, match='crosses'):
        resolve_positions(p, 'go')
    p = prepared()
    p.inputs['attention_mask'][0,1] = 0
    p.metadata['attention_mask'][1] = 0
    with pytest.raises(ValueError, match='padded'):
        resolve_positions(p, 'go')


def test_prefill_only_capture_does_not_modify_outputs_or_record_generation():
    o, p = owner(), prepared()
    positions = resolve_positions(p, 'go')
    baseline = o.model(**p.inputs).hidden_states[-1]
    capture = PositionCapture(o.modules, 3, p, positions)
    with capture:
        observed = o.model(**p.inputs).hidden_states[-1]
        saved = capture.stack('instruction_end').clone()
        for _, block in o.modules.values():
            block(torch.zeros(1,1,3))
    assert torch.equal(baseline, observed)
    assert torch.equal(saved, capture.stack('instruction_end'))
    assert capture.stack('prompt_end').shape == (2,3)
    assert len(capture.metadata()) == 4
    assert all(not block._forward_hooks for _,block in o.modules.values())


def test_reference_uses_final_pre_norm_and_equality_for_both_positions():
    o, p = owner(), prepared()
    positions = resolve_positions(p, 'go')
    references = independent_forward_check(o,p,positions)
    post_norm = o.model(**p.inputs).hidden_states[-1][0,1].detach()
    assert torch.allclose(references['instruction_end'][-1]*10, post_norm)
    assert not torch.equal(references['instruction_end'][-1], post_norm)
    capture, report = compare_capture(o,p,positions,lambda: o.model(**p.inputs))
    assert report['allclose']
    assert all(x['exact_equal'] for x in report['comparisons'].values())
    assert all(x['max_abs_difference'] == 0 for x in report['comparisons'].values())
    assert torch.equal(capture.stack('instruction_end'), references['instruction_end'])
    assert not o.model.model.language_model.norm._forward_pre_hooks
    assert all(not block._forward_hooks for _,block in o.modules.values())


def test_hook_cleanup_on_generation_and_shape_errors():
    o,p = owner(),prepared()
    positions = resolve_positions(p,'go')
    with pytest.raises(RuntimeError,match='failure'):
        with PositionCapture(o.modules,3,p,positions):
            raise RuntimeError('failure')
    with pytest.raises(ValueError,match='shape'):
        with PositionCapture(o.modules,3,p,positions):
            o.modules[0][1](torch.zeros(1,2,3))
    assert all(not block._forward_hooks for _,block in o.modules.values())


def test_independent_forward_error_removes_final_norm_hook():
    o,p = owner(),prepared()
    def fail(*args,**kwargs):
        raise RuntimeError('forward failure')
    o.model.forward = fail
    with pytest.raises(RuntimeError,match='forward failure'):
        independent_forward_check(o,p,resolve_positions(p,'go'))
    assert not o.model.model.language_model.norm._forward_pre_hooks
