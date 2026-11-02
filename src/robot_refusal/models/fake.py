"""Deterministic protocol fixture; no neural-model or semantic validity claims."""

from copy import deepcopy

from .base import GenerationResult, PreparedInput
from ..prompts import serialize
from ..records import CaptureConfig, InterventionConfig, ModelConfig


class FakeAdapter:
    backend = "fake"

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses
        self.index = 0
        self.closed = False

    def load(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self.index = 0
        self.closed = False

    def prepare(self, messages, generation_config) -> PreparedInput:
        serialized = serialize(messages)
        ids = list(serialized.encode("utf-8"))
        return PreparedInput({"messages": deepcopy(messages)}, {
            "serialized_prompt": serialized, "prompt_token_ids": ids,
            "attention_mask": [1] * len(ids), "token_encoding": "fake_utf8_bytes",
            "prompt_boundary": {"position": len(ids) - 1, "token_id": ids[-1]},
            "image_preprocessing": {"performed": False, "reason": "fake_backend"},
        }, generation_config)

    def generate(self, prepared, capture_config: CaptureConfig, intervention: InterventionConfig | None = None):
        if capture_config.enabled or intervention:
            raise ValueError("The fake adapter has no activations or intervention seam; use torch hook tests or Qwen")
        if self.responses is not None:
            if self.index >= len(self.responses):
                raise ValueError("Fake response script exhausted")
            response = self.responses[self.index]
        else:
            response = serialize({"assistant_text": "Protocol fixture: no physical action is performed.",
                                  "calls": [{"name": "finish", "arguments": {}}]})
        self.index += 1
        ids = list(response.encode("utf-8"))
        budget = prepared.generation_config["max_new_tokens"]
        truncated = len(ids) > budget
        ids = ids[:budget]
        return GenerationResult(bytes(ids).decode("utf-8", errors="replace"), ids,
            "length" if truncated else "stop", truncated,
            {"backend": "fake", "model_config": self.config.model_dump(), "model_commit": None,
             "is_real_model": False, "generation": prepared.generation_config})

    def close(self):
        self.closed = True
