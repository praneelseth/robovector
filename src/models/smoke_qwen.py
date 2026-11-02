"""TEMPORARY CPU smoke wrapper, not the research model-loading implementation.

Run with src/models/.venv/Scripts/python.exe -m src.eval.run_eval --config
configs/smoke.yaml. Weights and dependencies stay under this directory's ignored
.cache/ and .venv/. No robot execution occurs. Native Qwen tool templates are used.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"


def parse_tool_calls(text: str, tools: list[dict] | None) -> list[dict]:
    """Parse complete native XML calls (or JSON tags); never execute model text.

    Only advertised names and complete required arguments count as calls.
    Malformed/unrecognized tags remain in the raw text and produce a warning.
    """
    schemas = {t["function"]["name"]: t["function"] for t in tools or []}
    calls = []
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    for block in blocks:
        try:
            if block.strip().startswith("{"):
                obj = json.loads(block)
                name, arguments = obj["name"], obj["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            else:
                match = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block, re.S)
                if not match:
                    raise ValueError("Incomplete function tag")
                name, body = match.groups()
                arguments = {}
                for parameter, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
                    if parameter in arguments:
                        raise ValueError("Duplicate parameter")
                    value = value.strip()
                    kind = schemas.get(name, {}).get("parameters", {}).get("properties", {}).get(parameter, {}).get("type")
                    arguments[parameter] = value if kind == "string" else json.loads(value)
                if re.sub(r"<parameter=[^>]+>.*?</parameter>", "", body, flags=re.S).strip():
                    raise ValueError("Unparsed function content")
            if name not in schemas or not isinstance(arguments, dict):
                raise ValueError("Unknown tool or invalid arguments")
            params = schemas[name].get("parameters", {})
            if not set(params.get("required", [])) <= arguments.keys():
                raise ValueError("Missing required tool arguments")
            for key, value in arguments.items():
                if params.get("properties", {}).get(key, {}).get("type") == "string" and not isinstance(value, str):
                    raise ValueError("Expected string tool argument")
            calls.append({"id": f"smoke-call-{len(calls)}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
        except (ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"Unparsed tool call retained in output: {exc}")
    if text.count("<tool_call>") != len(blocks):
        warnings.warn("Truncated tool call retained in raw output")
    return calls


class SmokeQwen:
    """TEMPORARY ModelWrapper implementation for a small, real Qwen VLM."""

    def __init__(self, name: str = MODEL_ID):
        if name != MODEL_ID:
            raise ValueError(f"This temporary wrapper is pinned to {MODEL_ID}")
        self.name = name
        from huggingface_hub import snapshot_download, model_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        CACHE.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        snapshot_dir = CACHE / "model"
        marker = snapshot_dir / "smoke_revision.txt"
        if not marker.exists() or marker.read_text().strip() != REVISION:
            info = model_info(name, revision=REVISION, files_metadata=True)
            size = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
            if size > 5_000_000_000:
                raise RuntimeError(f"Download exceeds 5 GB ({size} bytes); get user confirmation before proceeding")
            print(f"Downloading pinned {name}: {size / 1e9:.2f} GB weights", flush=True)
            snapshot_download(name, revision=REVISION, local_dir=snapshot_dir,
                              allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "*.model", "LICENSE*"],
                              max_workers=2)
            marker.write_text(REVISION, encoding="utf-8")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        if self.device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("SMOKE_CPU_THREADS", "4")))
        self.processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot_dir, local_files_only=True, dtype=self.dtype, attn_implementation="eager")
        self.model.to(self.device).eval()
        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.d_model = self.model.config.text_config.hidden_size
        self.last_diagnostics = {}
        print(f"TEMPORARY {name}: device={self.device}, dtype={self.dtype}, expected shape={[self.n_layers, self.d_model]}", flush=True)

    def generate(self, messages: list[dict], images: list[Image.Image] | None,
                 tools: list[dict] | None, max_new_tokens: int,
                 temperature: float) -> dict:
        """Capture prompt residuals in one forward, then generate separately."""
        started = time.monotonic()
        native_messages = copy.deepcopy(messages)
        ordered_images = []
        for message in native_messages:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image":
                        index = item.pop("image_index")
                        if images is None or not 0 <= index < len(images):
                            raise ValueError("Image block has no corresponding PIL image")
                        ordered_images.append(images[index])
        prompt = self.processor.apply_chat_template(
            native_messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        if tools and any(t["function"]["name"] not in prompt for t in tools):
            raise ValueError("Native template did not include tool schemas")
        inputs = self.processor(text=[prompt], images=ordered_images or None,
                                return_tensors="pt", padding=True).to(self.device)
        last_index = int(inputs["attention_mask"][0].nonzero()[-1].item())
        final_residual = []

        def before_final_norm(module, args):
            final_residual.append(args[0][0, last_index].detach().float().cpu().clone())

        norm = self.model.model.language_model.norm
        hook = norm.register_forward_pre_hook(before_final_norm)
        with torch.inference_mode():
            try:
                forward = self.model(**inputs, output_hidden_states=True,
                                     use_cache=False, return_dict=True, logits_to_keep=1)
            finally:
                hook.remove()
            states = forward.hidden_states
            if len(states) != self.n_layers + 1 or len(final_residual) != 1:
                raise ValueError("Unexpected hidden-state layout; do not save misleading activations")
            # states[0] = embeddings; states[-1] = final normalized output.
            vectors = [state[0, last_index].detach().float().cpu() for state in states[1:-1]]
            hidden = torch.stack(vectors + final_residual).clone()
            del states, forward, vectors
            if hidden.shape != (self.n_layers, self.d_model) or not torch.isfinite(hidden).all():
                raise ValueError("Invalid last-prompt-token residuals")
            configured_eos = self.model.generation_config.eos_token_id
            configured_eos = configured_eos if isinstance(configured_eos, list) else [configured_eos]
            stop_ids = list(dict.fromkeys(t for t in [self.processor.tokenizer.eos_token_id, *configured_eos] if t is not None))
            options = dict(eos_token_id=stop_ids, pad_token_id=self.processor.tokenizer.pad_token_id,
                           max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                           use_cache=True, output_hidden_states=False, return_dict_in_generate=False)
            if temperature > 0:
                options.update(temperature=temperature, top_p=1.0, top_k=0)
            generated = self.model.generate(**inputs, **options)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        calls = parse_tool_calls(text, tools)
        eos_ids = stop_ids
        diagnostics = dict(model=self.name, revision=REVISION, device=str(self.device),
                           dtype=str(self.dtype), tool_mode="native_chat_template" if tools else "none",
                           prompt_tokens=inputs["input_ids"].shape[1], last_prompt_index=last_index,
                           generated_tokens=len(new_tokens), reached_token_limit=len(new_tokens) >= max_new_tokens,
                           ended_with_eos=bool(len(new_tokens) and new_tokens[-1].item() in eos_ids),
                           activation_shape=list(hidden.shape), elapsed_seconds=round(time.monotonic()-started, 2),
                           adapter_revision="smoke-v2", eos_token_ids=eos_ids,
                           raw_decoded_text=self.processor.decode(new_tokens, skip_special_tokens=False),
                           generated_token_ids=new_tokens.cpu().tolist(),
                           rendered_prompt=prompt, text=text, tool_calls=calls)
        self.last_diagnostics = diagnostics
        with (CACHE / "smoke_audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(diagnostics, ensure_ascii=False) + "\n")
        print(f"Smoke generation: {len(new_tokens)} tokens, {diagnostics['elapsed_seconds']}s, {len(calls)} tool calls", flush=True)
        return {"text": text, "tool_calls": calls, "hidden_states": hidden}


def create_model(name: str) -> SmokeQwen:
    """Factory matching the existing eval runner; no interface changes."""
    return SmokeQwen(name)
