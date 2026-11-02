"""Structural contracts implemented by the partner's adapters."""
from typing import Protocol

from PIL import Image


class ModelWrapper(Protocol):
    name: str

    def generate(self, messages: list[dict], images: list[Image.Image] | None,
                 tools: list[dict] | None, max_new_tokens: int,
                 temperature: float) -> dict:
        """Return text, tool_calls and [layers, width] last-prompt residuals.

        Residuals are captured BEFORE generation, after each transformer block,
        excluding the embedding layer and final normalization. Select the last
        non-padding prompt token after the model's multimodal/chat preprocessing.
        The adapter owns architecture-specific extraction and device handling.
        """
        ...


class RobotTools(Protocol):
    def schemas(self) -> list[dict]:
        """Return function tool definitions."""
        ...

    def execute(self, call: dict) -> dict:
        """Partner-owned execution; the evaluation driver never calls this."""
        ...
