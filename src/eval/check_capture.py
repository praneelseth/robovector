"""Compare generation-prefill hooks with an independent output_hidden_states pass.

Never downloads weights. The small cached model check is not an 8B validation.
"""
import argparse
import json
from pathlib import Path

import torch

from robot_refusal.activations import ResidualHooks
from robot_refusal.records import CaptureConfig


def compare_capture(model, inputs, modules, norm, *, generate, atol=1e-5, rtol=1e-5):
    model.eval()
    ids = inputs['input_ids'][0].tolist()
    mask = inputs['attention_mask'][0].tolist()
    boundary = max(i for i, value in enumerate(mask) if value)
    final = []
    def pre_norm(_module, args):
        final.append(args[0][0, boundary].detach().float().cpu().clone())
    handle = norm.register_forward_pre_hook(pre_norm)
    try:
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(output.hidden_states) != len(modules) + 1 or len(final) != 1:
        raise ValueError('Unexpected hidden-state layout')
    reference = torch.stack([state[0, boundary].detach().float().cpu()
                             for state in output.hidden_states[1:-1]] + final)
    del output
    hooks = ResidualHooks(modules, reference.shape[1], ids, mask, CaptureConfig(enabled=True))
    with torch.inference_mode(), hooks:
        generated = generate()
    sequences = generated.sequences if hasattr(generated, 'sequences') else generated
    hooks.finalize(sequences[0, len(ids):].tolist())
    observed = torch.stack([hooks.tensors[f'layer_{i}.prompt_{boundary}.residual'].float()
                            for i in range(len(modules))])
    report = dict(shape=list(reference.shape), prompt_boundary=boundary,
        dtype=str(next(model.parameters()).dtype), device=str(next(model.parameters()).device),
        exact_equal=torch.equal(reference, observed),
        allclose=torch.allclose(reference, observed, atol=atol, rtol=rtol), atol=atol, rtol=rtol,
        max_abs_difference=(reference - observed).abs().max().item(),
        per_layer_max_abs_difference=(reference - observed).abs().amax(dim=1).tolist(),
        reference='output_hidden_states=True, separate use_cache=False forward; final pre-norm vector',
        observed='Praneel ResidualHooks during generation prefill; decoder block outputs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-smoke', action='store_true')
    parser.add_argument('--model-dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if args.cached_smoke:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from src.models.smoke_qwen import CACHE, REVISION, MODEL_ID
        path = CACHE / 'model'
        if (path / 'smoke_revision.txt').read_text().strip() != REVISION:
            raise ValueError('Pinned small-model cache is missing')
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
                    dtype=torch.float32, attn_implementation='eager').eval()
        messages = [{'role': 'user', 'content': 'Describe how to move an empty cup onto a counter.'}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[prompt], return_tensors='pt')
        layers = model.model.language_model.layers
        modules = {i: (f'model.language_model.layers.{i}', layer) for i, layer in enumerate(layers)}
        identity = dict(model=MODEL_ID, revision=REVISION, validation_scope='cached small model, text only')
    else:
        from robot_refusal.models.qwen_vl import QwenVLAdapter
        from robot_refusal.records import ModelConfig
        adapter = QwenVLAdapter()
        adapter.load(ModelConfig(model_dir=args.model_dir, device=args.device))
        model, processor, modules = adapter.model, adapter.processor, adapter.modules
        content = ([{'type': 'image', 'image': str(args.image.resolve())}] if args.image else [])
        content += [{'type': 'text', 'text': 'Describe the scene briefly.'}]
        prepared = adapter.prepare([{'role': 'user', 'content': content}], {})
        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in prepared.inputs.items()}
        identity = dict(model=adapter.config.model_id, revision=adapter.snapshot['model_commit'],
                        validation_scope='Qwen3-VL checkpoint', image=str(args.image) if args.image else None)
    eos = model.generation_config.eos_token_id
    report = compare_capture(model, inputs, modules, model.model.language_model.norm,
        generate=lambda: model.generate(**inputs, max_new_tokens=2, do_sample=False,
            use_cache=True, pad_token_id=eos[0] if isinstance(eos, list) else eos))
    report.update(identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['allclose']:
        raise SystemExit('Capture comparison failed; do not treat the methods as interchangeable')


if __name__ == '__main__':
    main()
