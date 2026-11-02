"""Offline checks for the temporary adapter; no model downloads."""
import json
from types import SimpleNamespace

import pytest
import torch

import src.models.smoke_qwen as smoke
from src.stubs.robot_tools import RobotTools


def test_native_tool_parsing_and_incomplete_calls():
    schemas = RobotTools().schemas()
    text = '<tool_call><function=move_object><parameter=object>can</parameter><parameter=destination>counter</parameter></function></tool_call>'
    calls = smoke.parse_tool_calls(text, schemas)
    assert json.loads(calls[0]['function']['arguments']) == {'object': 'can', 'destination': 'counter'}
    with pytest.warns(UserWarning):
        assert smoke.parse_tool_calls('<tool_call>unfinished', schemas) == []
    with pytest.warns(UserWarning):
        assert smoke.parse_tool_calls(text.replace('move_object', 'unknown'), schemas) == []
    assert smoke.parse_tool_calls('I cannot help.', schemas) == []


def test_prompt_layer_selection_and_separate_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'CACHE', tmp_path)
    events = []
    # Two layers, three token positions, two features. Last real token is index 1.
    embedding = torch.zeros(1, 3, 2)
    layer_one = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    layer_two = layer_one + 10
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs['enable_thinking'] is False
            return 'rendered prompt'
        def __call__(self, **kwargs):
            return Batch(input_ids=torch.tensor([[4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 0]]))
        def decode(self, tokens, **kwargs):
            return 'Done.'
    class Model:
        def __init__(self):
            self.norm = torch.nn.Identity()
            self.model = SimpleNamespace(language_model=SimpleNamespace(norm=self.norm))
            self.generation_config = SimpleNamespace(eos_token_id=9)
        def __call__(self, **kwargs):
            events.append('prompt_forward')
            assert kwargs['output_hidden_states'] is True
            assert kwargs['use_cache'] is False
            self.norm(layer_two)
            return SimpleNamespace(hidden_states=(embedding, layer_one, layer_two * 100))
        def generate(self, **kwargs):
            events.append('generate')
            assert not self.norm._forward_pre_hooks
            assert kwargs['output_hidden_states'] is False
            assert kwargs['eos_token_id'] == [9]
            return torch.tensor([[4, 5, 0, 8, 9]])
    wrapper = smoke.SmokeQwen.__new__(smoke.SmokeQwen)
    wrapper.name = smoke.MODEL_ID
    wrapper.device = torch.device('cpu')
    wrapper.dtype = torch.float32
    wrapper.n_layers, wrapper.d_model = 2, 2
    wrapper.model, wrapper.processor = Model(), Processor()
    result = wrapper.generate([{'role': 'user', 'content': 'test'}], None, None, 256, 0.7)
    assert events == ['prompt_forward', 'generate']
    assert torch.equal(result['hidden_states'], torch.tensor([[3., 4.], [13., 14.]]))
    assert result['text'] == 'Done.'
    assert result['tool_calls'] == []
