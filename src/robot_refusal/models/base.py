from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..records import CaptureConfig, InterventionConfig, ModelConfig


@dataclass
class PreparedInput:
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    generation_config: dict[str, Any]


@dataclass
class GenerationResult:
    raw_text: str
    generated_ids: list[int]
    finish_reason: str
    token_limit_reached: bool
    resolved_configuration: dict[str, Any]
    activations: dict[str, Any] = field(default_factory=dict)
    activation_metadata: list[dict[str, Any]] = field(default_factory=list)
    unavailable_activations: list[dict[str, Any]] = field(default_factory=list)
    next_token_statistics: dict[str, Any] | None = None
    capture_error: str | None = None


class ModelAdapter(Protocol):
    backend: str

    def load(self, model_config: ModelConfig) -> None: ...
    def prepare(self, messages: list[dict[str, Any]], generation_config: dict[str, Any]) -> PreparedInput: ...
    def generate(self, prepared: PreparedInput, capture_config: CaptureConfig,
                 intervention: InterventionConfig | None = None) -> GenerationResult: ...
    def close(self) -> None: ...
