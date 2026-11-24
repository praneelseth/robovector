"""Experimental two-position prefill capture, independent of production interfaces.

No downloads or model loading. Caller supplies an already loaded Qwen adapter and
its PreparedInput. Intended for text-only, batch-one, interventions-off diagnostics.
"""
from __future__ import annotations

import torch


def _prompt(prepared):
    ids, mask = prepared.inputs['input_ids'], prepared.inputs['attention_mask']
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError('Require batch-one input IDs and matching attention mask')
    token_ids, attention = ids[0].tolist(), mask[0].tolist()
    if not token_ids or not any(attention) or any(x not in (0, 1) for x in attention):
        raise ValueError('Require nonempty binary attention mask')
    if (token_ids != prepared.metadata['prompt_token_ids'] or
            attention != prepared.metadata['attention_mask']):
        raise ValueError('Prepared tensors and token metadata disagree')
    return token_ids, attention


def _validate_positions(prepared, positions):
    ids, mask = _prompt(prepared)
    if set(positions) != {'instruction_end', 'prompt_end'}:
        raise ValueError('Require instruction_end and prompt_end positions')
    for name, index in positions.items():
        if type(index) is not int or not 0 <= index < len(ids) or not mask[index]:
            raise ValueError(f'Invalid or padded {name} position')
    boundary = max(i for i, value in enumerate(mask) if value)
    if positions['prompt_end'] != boundary or positions['instruction_end'] > boundary:
        raise ValueError('Positions disagree with actual last nonpadding token')
    return ids, mask


def resolve_positions(prepared, instruction):
    """Resolve the last instruction character against full-prompt offset mapping.

    Ambiguous occurrences, missing/overlapping offsets, padding, and mismatched
    metadata fail. Whitespace after an instruction is not an instruction token.
    No standalone tokenization or fixed suffix-length assumptions are used.
    """
    ids, mask = _prompt(prepared)
    text = prepared.metadata['expanded_prompt']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Require a nonempty instruction')
    if text.count(instruction) != 1:
        raise ValueError('Instruction must occur exactly once in expanded prompt')
    start = text.index(instruction)
    end = start + len(instruction.rstrip())
    offsets = prepared.metadata['token_offsets_in_processor_expanded_text']
    if len(offsets) != len(ids):
        raise ValueError('Offset/token length mismatch')
    candidates = []
    for index, offset in enumerate(offsets):
        if (not isinstance(offset, (list, tuple)) or len(offset) != 2 or
                any(type(v) is not int for v in offset)):
            raise ValueError('Invalid offset pair')
        left, right = offset
        if not 0 <= left <= right <= len(text):
            raise ValueError('Offset outside expanded prompt')
        if left <= end - 1 < right:
            # A token straddling instruction/end-template is not a precise endpoint.
            if left < start or right > end:
                raise ValueError('Final instruction token crosses instruction span')
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError('Instruction endpoint is missing or ambiguous in token offsets')
    positions = dict(instruction_end=candidates[0],
                     prompt_end=max(i for i, value in enumerate(mask) if value))
    _validate_positions(prepared, positions)
    return positions


class PositionCapture:
    """Copy two block-output vectors during first invocation; never modify output."""
    def __init__(self, modules, hidden_size, prepared, positions):
        self.ids, self.mask = _validate_positions(prepared, positions)
        if set(modules) != set(range(len(modules))) or not modules or hidden_size < 1:
            raise ValueError('Require contiguous decoder modules and a hidden size')
        self.modules, self.hidden_size = modules, hidden_size
        self.positions, self.tensors, self.handles = dict(positions), {}, []
        self.invocations = {layer: 0 for layer in modules}

    def __enter__(self):
        if self.handles or any(self.invocations.values()):
            raise RuntimeError('Capture context is single-use')
        try:
            for layer in sorted(self.modules):
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

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            invocation = self.invocations[layer]
            self.invocations[layer] += 1
            if invocation:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if (not isinstance(hidden, torch.Tensor) or
                    tuple(hidden.shape) != (1, len(self.ids), self.hidden_size)):
                raise ValueError('Unexpected prefill block-output shape')
            for name, index in self.positions.items():
                vector = hidden[0, index].detach().cpu().clone()
                if not vector.is_floating_point() or not torch.isfinite(vector).all():
                    raise ValueError('Invalid captured vector')
                self.tensors[(name, layer)] = vector
            return None
        return hook

    def stack(self, name):
        if name not in self.positions or any((name, layer) not in self.tensors for layer in self.modules):
            raise ValueError('Missing named prefill layer captures')
        return torch.stack([self.tensors[(name, layer)] for layer in sorted(self.modules)])

    def metadata(self):
        return [dict(name=name, layer=layer, module=self.modules[layer][0],
                     sequence_position=index, token_id=self.ids[index], stage='prefill',
                     residual_location='decoder_block_output', phase='unmodified')
                for name, index in self.positions.items() for layer in sorted(self.modules)]


def independent_forward_check(owner, prepared, positions):
    """Return two-position references using hidden states + final pre-norm hook.

    The adapter must be Qwen-like with model.model.language_model.norm. This is an
    extra diagnostic pass; ordinary two-position capture needs no extra pass.
    """
    _validate_positions(prepared, positions)
    owner.model.eval()
    final = []
    def before_norm(_module, args):
        hidden = args[0]
        final.append({name: hidden[0, index].detach().float().cpu().clone()
                      for name, index in positions.items()})
    norm = owner.model.model.language_model.norm
    handle = norm.register_forward_pre_hook(before_norm)
    try:
        device = owner.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in prepared.inputs.items()}
        with torch.inference_mode():
            output = owner.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(output.hidden_states) != len(owner.modules) + 1 or len(final) != 1:
        raise ValueError('Unexpected independent hidden-state layout')
    references = {}
    for name, index in positions.items():
        vectors = [state[0, index].detach().float().cpu() for state in output.hidden_states[1:-1]]
        references[name] = torch.stack(vectors + [final[0][name]])
        if (references[name].shape != (len(owner.modules), owner.hidden_size) or
                not torch.isfinite(references[name]).all()):
            raise ValueError('Invalid independent reference tensor')
    return references


def compare_capture(owner, prepared, positions, generate, *, atol=1e-5, rtol=1e-5):
    """Run diagnostic forward then caller's generation; return capture and report.

    Caller supplies interventions-off generation. Does not alter RNG or settings;
    caller should reset the intended generation seed after the reference if needed.
    """
    references = independent_forward_check(owner, prepared, positions)
    capture = PositionCapture(owner.modules, owner.hidden_size, prepared, positions)
    with torch.inference_mode(), capture:
        generate()
    comparisons = {}
    for name in positions:
        observed = capture.stack(name).float()
        delta = (references[name] - observed).abs()
        comparisons[name] = dict(shape=list(observed.shape), exact_equal=torch.equal(references[name], observed),
            allclose=torch.allclose(references[name], observed, atol=atol, rtol=rtol),
            max_abs_difference=delta.max().item(), per_layer_max_abs_difference=delta.amax(dim=1).tolist())
    return capture, dict(positions=dict(positions), atol=atol, rtol=rtol,
                         allclose=all(row['allclose'] for row in comparisons.values()),
                         comparisons=comparisons,
                         reference='Separate output_hidden_states forward; final vector before final norm',
                         observed='Generation-prefill decoder block outputs; no interventions')
