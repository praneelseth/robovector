"""Real Qwen3-VL architecture with tiny random weights; no checkpoint download."""
import torch
import pytest
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from src.eval.check_capture import compare_capture


@pytest.mark.parametrize('vision', [False, True])
def test_qwen3vl_generation_prefill_matches_separate_forward(vision):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            out_hidden_size=32, deepstack_visual_indexes=[0]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    config._attn_implementation = 'eager'
    model = Qwen3VLForConditionalGeneration(config).eval()
    inputs = {'input_ids': torch.tensor([[3, 4, 5, 6]]), 'attention_mask': torch.ones(1, 4, dtype=torch.long)}
    if vision:
        inputs = {'input_ids': torch.tensor([[62, 60, 60, 60, 60, 63, 3, 4]]),
                  'attention_mask': torch.ones(1, 8, dtype=torch.long),
                  'pixel_values': torch.randn(4, 12), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    modules = {i: (f'model.language_model.layers.{i}', layer)
               for i, layer in enumerate(model.model.language_model.layers)}
    report = compare_capture(model, inputs, modules, model.model.language_model.norm,
        generate=lambda: model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                         use_cache=True, pad_token_id=0, eos_token_id=2))
    assert report['shape'] == [2, 32]
    assert report['exact_equal'], report
    assert report['max_abs_difference'] == 0
