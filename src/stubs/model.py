"""Deterministic under torch.manual_seed; outputs are synthetic, not findings."""
import torch


class StubModel:
    def __init__(self, name: str = "stub", n_layers: int = 32, d_model: int = 4096):
        self.name = name
        self.n_layers = n_layers
        self.d_model = d_model

    def generate(self, messages, images, tools, max_new_tokens, temperature):
        if tools:
            text = "I will attempt the requested movement."
            calls = [{"id": "stub-call", "type": "function", "function": {
                "name": "move_object", "arguments": '{"object": "requested object", "destination": "requested destination"}'}}]
        else:
            text = "I cannot help with that request."
            calls = []
        return {"text": text, "tool_calls": calls,
                "hidden_states": torch.randn(self.n_layers, self.d_model)}


def create_model(name: str):
    """Factory entry point used by YAML/CLI; name is a recorded label only."""
    return StubModel(name)
