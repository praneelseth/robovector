# Temporary Qwen smoke wrapper

This directory is a disposable bridge until the partner's ModelWrapper is ready.
It implements the existing generate signature without changing the eval runner.

Hardware checked: Intel i7-1360P, Iris Xe integrated graphics, 31.6 GiB system RAM;
no CUDA or Apple MPS. Model: Qwen/Qwen3.5-0.8B (post-trained vision-language model,
not the base checkpoint), pinned to 2fc06364715b967f1860aea9cf38778875588b17.
Official source: https://huggingface.co/Qwen/Qwen3.5-0.8B
Weights: 1,746,942,600 bytes (~1.75 GB). Cache and venv are ignored here.

Setup (reuses installed CPU PyTorch):

```powershell
python -m venv --system-site-packages src/models/.venv
src/models/.venv/Scripts/python -m pip install transformers==5.22.1 huggingface_hub==2.4.0 safetensors==0.8.0
src/models/.venv/Scripts/python -m src.eval.run_eval --config configs/smoke.yaml
```

The config runs burner/burner_safe, text_chat/text_tools, two repeats, temperature
0.7, max_new_tokens 256, seed 42. CPU uses float32 and four Torch threads (override
SMOKE_CPU_THREADS). Generation uses top_p=1 and top_k=0, with thinking disabled.
This is not equivalent to the requested 8B research model or to partner defaults.

Tool schemas go directly into the native chat template (`tools=...`). The wrapper
parses Qwen3.5's complete <tool_call><function=...><parameter=...> blocks and also
accepts JSON tool-call blocks. Only advertised tools with required arguments count.
Raw outputs retain tags and malformed calls; no tool is executed. RobotTools still
supplies the project's stub move_object schema.

One prompt-only forward uses output_hidden_states=True, use_cache=False and
logits_to_keep=1. We drop embeddings, select the last non-padding prompt token,
and replace Transformers' final normalized output with the final norm's input
captured during that same forward. Result: [24, 1024], float32 CPU, after each
language decoder block and before final normalization. Generation is a separate
call, with no prompt cache reused. This hybrid Qwen3.5 backbone contains both
linear-attention and full-attention blocks; layers are not Qwen3-VL-8B layers.

The normal runner saves responses, labels and activations in outputs/. Additional
audit records in `.cache/smoke_audit.jsonl` retain rendered prompts, model revision,
token counts, truncation flags and elapsed time; each record corresponds to one
generate call. Inspect these when a rule-based label disagrees with the response.
The wrapper is pinned and fails rather than silently switching models. Its download
guard rejects weights exceeding 5 GB. Images are supported via the existing image
indices, but this smoke configuration exercises text conditions only.

If the Hub client stalls on the weight transfer, run `src/models/.venv/Scripts/python -m src.models.download_checkpoint` after the small metadata/tokenizer files have downloaded. This range downloader resumes verified-length chunks and validates the complete checkpoint against the pinned Hugging Face SHA-256 before marking it ready.

Generation explicitly stops at either the tokenizer chat EOS (`<|im_end|>`, 248046) or the checkpoint EOS (`<|endoftext|>`, 248044). The checkpoint has no generation_config.json and its text config alone would stop only at endoftext. The initial diagnostic run exposed stray conversation text; smoke-v2 fixes this and retains raw token IDs and special-token decoding in its audit. Treat the initial diagnostic run as superseded.

Final validated report: [SMOKE_RESULTS.md](SMOKE_RESULTS.md). To regenerate it for the latest completed smoke-v2 run: `src/models/.venv/Scripts/python -m src.models.report_smoke outputs/runs/<run-directory>`. This checks all eight saved tensors, repeat equality and matching audit records before writing the report.
