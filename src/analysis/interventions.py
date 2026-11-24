"""Experimental decoder-block-output interventions for controlled steering pilots.

This adapts interventions to partner decoder-block outputs, not resid_pre or
embeddings. Hooks operate on every sequence position in both prefill and cached
decode. The caller owns model loading, capture-hook order, and provenance.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def _direction(vector):
    if (not isinstance(vector, torch.Tensor) or vector.ndim != 1 or vector.numel() == 0
            or not vector.is_floating_point() or not torch.isfinite(vector).all()):
        raise ValueError('Direction must be a nonempty finite floating 1D tensor')
    copy = vector.detach().cpu().clone()
    norm = torch.linalg.vector_norm(copy.double()).item()
    if norm == 0 or not math.isfinite(norm):
        raise ValueError('Direction must have finite nonzero L2 norm')
    return copy, norm


class AllPositionIntervention:
    """Addition at one block, or projection ablation at one/all blocks.

    ``modules`` maps nonnegative integer IDs to ``(name, torch.nn.Module)``.
    Addition: ``h += strength * direction``. Ablation:
    ``h -= strength * dot(h, unit_direction) * unit_direction``.
    A zero strength is an exact sham: hooks never replace or mutate outputs.
    All positions are changed; no token-index selection or generation changes.
    """
    def __init__(self, modules, direction, mode='addition', layer=None, strength=1):
        self.direction, self.direction_norm = _direction(direction)
        if mode not in ('addition', 'ablation'):
            raise ValueError('Mode must be addition or ablation')
        if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength):
            raise ValueError('Strength must be a finite real number')
        if not isinstance(modules, dict) or not modules:
            raise ValueError('Require a nonempty decoder-module mapping')
        for index, item in modules.items():
            if (type(index) is not int or index < 0 or not isinstance(item, tuple) or len(item) != 2
                    or not isinstance(item[0], str) or not item[0]
                    or not isinstance(item[1], torch.nn.Module)):
                raise ValueError('Invalid decoder-module ID/name/module')
        if len({id(item[1]) for item in modules.values()}) != len(modules):
            raise ValueError('Decoder IDs must identify distinct modules')
        if layer is not None and (type(layer) is not int or layer not in modules):
            raise ValueError('Unknown decoder layer')
        if mode == 'addition' and layer is None:
            raise ValueError('Addition requires one explicit decoder layer')
        self.mode, self.layer, self.strength = mode, layer, float(strength)
        self.modules = dict(modules)
        self.target_layers = sorted(modules) if layer is None else [layer]
        self.handles = []
        self._entered = False
        # Normalize once on CPU, then transfer at most once per device/dtype in
        # this context. Avoid a host-to-device copy at every block/decode token.
        self.unit_direction = self.direction.double() / self.direction_norm
        self._device_cache = {}

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Intervention context is single-use')
        self._entered = True
        try:
            for layer in self.target_layers:
                self.handles.append(self.modules[layer][1].register_forward_hook(self._hook))
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._device_cache.clear()

    def __exit__(self, *_):
        self.close()

    def _converted_direction(self, device, dtype):
        key = (device, dtype)
        if key not in self._device_cache:
            source = self.direction if self.mode == 'addition' else self.unit_direction
            self._device_cache[key] = source.to(device=device, dtype=dtype)
        return self._device_cache[key]

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) and output else output
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or
                hidden.shape[-1] != self.direction.numel() or not hidden.is_floating_point()):
            raise ValueError('Decoder output must be floating [batch, sequence, direction_width]')
        if self.strength == 0:
            return None
        if self.mode == 'addition':
            direction = self._converted_direction(hidden.device, hidden.dtype)
            modified = hidden + self.strength * direction
        else:
            # Compute half-precision projections in float32, preserving output dtype.
            dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            unit = self._converted_direction(hidden.device, dtype)
            values = hidden.to(dtype=dtype)
            projection = (values * unit).sum(dim=-1, keepdim=True) * unit
            modified = (values - self.strength * projection).to(dtype=hidden.dtype)
        return (modified, *output[1:]) if isinstance(output, tuple) else modified


def seeded_random_direction(reference, seed):
    """Deterministic CPU Gaussian direction with the reference's L2 norm/dtype."""
    reference, norm = _direction(reference)
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise ValueError('Seed must be an integer in [0, 2**63)')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    vector = torch.randn(reference.shape, generator=generator, dtype=torch.float64)
    vector *= norm / torch.linalg.vector_norm(vector)
    return vector.to(dtype=reference.dtype)
