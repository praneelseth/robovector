"""Targeted decoder block-output hooks for batch-one cached generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .records import CaptureConfig, InterventionConfig


class CaptureError(RuntimeError):
    pass


class ResidualHooks:
    """First invocation is prefill, even for a one-token prompt.

    Subsequent one-token invocations contextualize generated tokens. Merely sampling
    the last token does not produce a residual for it. No extra forward pass is run.
    """

    def __init__(self, modules: dict[int, tuple[str, torch.nn.Module]], hidden_size: int,
                 prompt_ids: list[int], attention_mask: list[int], capture: CaptureConfig,
                 intervention: InterventionConfig | None = None):
        self.modules = modules
        self.hidden_size = hidden_size
        self.prompt_ids = prompt_ids
        self.prompt_length = len(prompt_ids)
        if len(attention_mask) != self.prompt_length or not any(attention_mask):
            raise CaptureError("Expected a nonempty batch-one prompt and matching attention mask")
        self.boundary = max(i for i, value in enumerate(attention_mask) if value)
        self.capture = capture
        self.capture_layers = set(modules if capture.layers is None else capture.layers) if capture.enabled else set()
        self.intervention = intervention
        self.selected_layers = self.capture_layers | ({intervention.layer} if intervention else set())
        if not self.selected_layers.issubset(modules):
            raise CaptureError(f"Unknown decoder layers: {self.selected_layers - modules.keys()}")
        self.direction = None
        self.direction_sha256 = None
        if intervention:
            path = Path(intervention.direction_path)
            self.direction_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.direction = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(self.direction, torch.Tensor) or self.direction.shape != (hidden_size,):
                raise CaptureError(f"Direction must be a single tensor with shape ({hidden_size},)")
            if not self.direction.is_floating_point() or not torch.isfinite(self.direction).all():
                raise CaptureError("Direction must contain finite floating-point values")
        self.handles = []
        self.invocations = {layer: 0 for layer in self.selected_layers}
        self.tensors = {}
        self.metadata = []
        self.unavailable = []

    def __enter__(self):
        try:
            for layer in sorted(self.selected_layers):
                self.handles.append(self.modules[layer][1].register_forward_hook(self._hook(layer)))
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *_):
        self.close()

    def _save(self, layer, hidden, position, stage, generated_index, phase):
        index = self.boundary if stage == "prompt" else generated_index
        key = f"layer_{layer}.{stage}_{index}.{phase}"
        tensor = hidden[0, position, :].detach().to("cpu").clone()
        self.tensors[key] = tensor
        self.metadata.append({"key": key, "layer": layer, "module": self.modules[layer][0],
            "residual_location": "decoder_block_output", "stage": stage, "phase": phase,
            "sequence_position": self.boundary if stage == "prompt" else self.prompt_length + generated_index,
            "generated_token_index": generated_index,
            "token_id": self.prompt_ids[self.boundary] if stage == "prompt" else None,
            "dtype": str(tensor.dtype), "shape": list(tensor.shape)})

    def _hook(self, layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise CaptureError(f"{self.modules[layer][0]} must output [batch, sequence, hidden]")
            if hidden.shape[0] != 1 or hidden.shape[2] != self.hidden_size:
                raise CaptureError(f"Unexpected residual dimensions: {tuple(hidden.shape)}")
            invocation = self.invocations[layer]
            stage = "prompt" if invocation == 0 else "generation"
            expected_length = self.prompt_length if invocation == 0 else 1
            if hidden.shape[1] != expected_length:
                raise CaptureError(f"Expected {stage} sequence length {expected_length}, got {hidden.shape[1]}")
            self.invocations[layer] += 1
            generated_index = invocation - 1 if invocation else None
            position = self.boundary if stage == "prompt" else 0
            capture_here = layer in self.capture_layers and (
                stage == "prompt" or generated_index in self.capture.generated_token_indices)
            intervene_here = self.intervention is not None and layer == self.intervention.layer and (
                stage == "prompt" or self.intervention.schedule == "prompt_boundary_and_generation")
            if capture_here or intervene_here:
                self._save(layer, hidden, position, stage, generated_index,
                           "pre_intervention" if intervene_here else "residual")
            modified = None
            if intervene_here:
                # A zero-strength intervention does not even replace the output tensor.
                if self.intervention.strength != 0:
                    modified = hidden.clone()
                    direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
                    modified[:, position, :] = hidden[:, position, :] + self.intervention.strength * direction
                self._save(layer, hidden if modified is None else modified, position, stage,
                           generated_index, "post_intervention")
            if modified is not None:
                return (modified, *output[1:]) if isinstance(output, tuple) else modified
            return None
        return hook

    def finalize(self, generated_ids: list[int]):
        """Verify required captures; explicitly mark positions never forwarded."""
        for layer in self.selected_layers:
            if self.invocations[layer] == 0:
                raise CaptureError(f"Configured decoder layer {layer} was never called")
        for row in self.metadata:
            index = row["generated_token_index"]
            if index is not None:
                if index >= len(generated_ids):
                    raise CaptureError("Captured a generated position with no corresponding output token")
                row["token_id"] = generated_ids[index]
        for layer in self.capture_layers:
            for index in self.capture.generated_token_indices:
                found = any(row["layer"] == layer and row["generated_token_index"] == index for row in self.metadata)
                if not found:
                    if index < len(generated_ids) - 1:
                        raise CaptureError(f"Missing forwarded generated token {index} at layer {layer}")
                    self.unavailable.append({"layer": layer, "generated_token_index": index,
                        "reason": "not_generated" if index >= len(generated_ids) else "sampled_but_not_forwarded"})
