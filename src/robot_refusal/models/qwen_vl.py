"""Offline Qwen3-VL adapter using the snapshot's official processor/template."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .base import GenerationResult, PreparedInput
from ..activations import CaptureError, ResidualHooks
from ..model_setup import read_snapshot
from ..records import CaptureConfig, InterventionConfig, ModelConfig


class QwenVLAdapter:
    backend = "qwen"

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self, model_config: ModelConfig):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path, self.snapshot = read_snapshot(model_config.model_dir, model_config.model_id)
        self.config = model_config
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
            use_fast=True, min_pixels=model_config.image_min_pixels, max_pixels=model_config.image_max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=getattr(torch, model_config.dtype),
            device_map=model_config.device, attn_implementation=model_config.attn_implementation)
        self.model.eval()
        # Pinned Transformers 4.57.1: block outputs precede the parent's DeepStack
        # injection at early visual positions, and precede final language-model norm.
        layers = self.model.model.language_model.layers
        text_config = self.model.config.text_config
        if len(layers) != text_config.num_hidden_layers:
            raise CaptureError("Language decoder layer count differs from the model configuration")
        self.hidden_size = text_config.hidden_size
        self.modules = {i: (f"model.language_model.layers.{i}", layer) for i, layer in enumerate(layers)}
        for name, module in self.modules.values():
            if self.model.get_submodule(name) is not module or type(module).__name__ != "Qwen3VLTextDecoderLayer":
                raise CaptureError(f"Unexpected Qwen decoder module at {name}")
        self.resolved = {"backend": self.backend, "model_commit": self.snapshot["model_commit"],
            "snapshot": self.snapshot, "model_config": model_config.model_dump(),
            "actual_dtype": str(self.model.dtype),
            "device_map": {k: str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
            "decoder_modules": {str(i): name for i, (name, _) in self.modules.items()},
            "hidden_size": self.hidden_size,
            "measurement_point": "decoder block output, before parent DeepStack injection and final norm",
            "deepstack_visual_indexes": self.model.config.vision_config.deepstack_visual_indexes,
            "checkpoint_generation_defaults": self.model.generation_config.to_dict(),
            "quantization": None}

    def prepare(self, messages, generation_config):
        if self.model is None or self.processor is None:
            raise RuntimeError("load() must precede prepare()")
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", return_offsets_mapping=True,
            images_kwargs={"min_pixels": self.config.image_min_pixels, "max_pixels": self.config.image_max_pixels})
        offsets = inputs.pop("offset_mapping")
        inputs.pop("token_type_ids", None)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
            raise ValueError("Only batch size one with a matching attention mask is supported")
        token_ids = ids[0].tolist()
        attention_mask = mask[0].tolist()
        boundary = max(i for i, value in enumerate(attention_mask) if value)
        image_paths = [part["image"] for message in messages for part in message["content"] if part["type"] == "image"]
        image_processor = self.processor.image_processor
        patch_size = image_processor.patch_size
        merge_size = image_processor.merge_size
        grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        if len(grids) != len(image_paths):
            raise ValueError("Processor image grids do not match the supplied images")
        positions = [i for i, token in enumerate(token_ids) if token == self.processor.image_token_id]
        image_metadata = []
        offset = 0
        for path, grid in zip(image_paths, grids):
            count = grid[0] * grid[1] * grid[2] // (merge_size**2)
            layout = positions[offset:offset + count]
            if len(layout) != count:
                raise ValueError("Image token layout does not match image_grid_thw")
            with Image.open(path) as image:
                original_size = list(image.size)
                oriented_size = list(ImageOps.exif_transpose(image).size)
            image_metadata.append({"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "original_size_wh": original_size, "oriented_size_wh": oriented_size,
                "resized_size_wh": [grid[2] * patch_size, grid[1] * patch_size],
                "image_grid_thw": grid, "visual_token_count": count, "visual_token_positions": layout})
            offset += count
        if offset != len(positions):
            raise ValueError("Unaccounted image tokens in formatted input")
        template = self.processor.chat_template
        template_string = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
        metadata = {"serialized_prompt": rendered,
            "expanded_prompt": self.processor.tokenizer.decode(token_ids, skip_special_tokens=False,
                                                                clean_up_tokenization_spaces=False),
            "prompt_token_ids": token_ids, "attention_mask": attention_mask,
            "token_pieces": self.processor.tokenizer.convert_ids_to_tokens(token_ids),
            "token_offsets_in_processor_expanded_text": offsets[0].tolist(),
            "prompt_boundary": {"position": boundary, "token_id": token_ids[boundary]},
            "chat_template": template, "chat_template_sha256": hashlib.sha256(template_string.encode()).hexdigest(),
            "image_preprocessing": {"images": image_metadata, "patch_size": patch_size, "merge_size": merge_size,
                "temporal_patch_size": image_processor.temporal_patch_size,
                "processor_class": type(image_processor).__name__, "settings": image_processor.to_dict(),
                "min_pixels": self.config.image_min_pixels, "max_pixels": self.config.image_max_pixels},
            "input_tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                              for name, tensor in inputs.items()},
            "observation_updated": False}
        return PreparedInput(dict(inputs), metadata, generation_config)

    def generate(self, prepared: PreparedInput, capture_config: CaptureConfig,
                 intervention: InterventionConfig | None = None):
        from transformers import GenerationConfig as HFGenerationConfig

        settings = prepared.generation_config
        sample = settings["decoding"] == "fixed_temperature"
        original = self.model.generation_config
        eos = original.eos_token_id
        pad = original.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        # Use an explicit neutral reference config; do not inherit sampling penalties
        # or temperature from the checkpoint's recommended chat configuration.
        effective = HFGenerationConfig(max_new_tokens=settings["max_new_tokens"], do_sample=sample,
            temperature=settings["temperature"] if sample else 1.0,
            top_p=settings["top_p"] if sample else 1.0, top_k=settings["top_k"] if sample else 0,
            repetition_penalty=settings["repetition_penalty"], num_beams=1, num_return_sequences=1,
            use_cache=True, bos_token_id=original.bos_token_id, eos_token_id=eos, pad_token_id=pad,
            return_dict_in_generate=True, output_scores=False, output_logits=False)
        device = self.model.get_input_embeddings().weight.device
        if device.type == "meta":
            raise RuntimeError("Input embeddings are offloaded to meta; choose a device with sufficient memory")
        inputs = {name: tensor.to(device) for name, tensor in prepared.inputs.items()}
        hooks = ResidualHooks(self.modules, self.hidden_size, prepared.metadata["prompt_token_ids"],
                              prepared.metadata["attention_mask"], capture_config, intervention)
        cuda_devices = sorted({parameter.device.index for parameter in self.model.parameters()
                               if parameter.device.type == "cuda"})
        for index in cuda_devices:
            torch.cuda.reset_peak_memory_stats(index)
        with torch.inference_mode(), hooks:
            output = self.model.generate(**inputs, generation_config=effective)
        ids = output.sequences[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
        raw = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos_ids = eos if isinstance(eos, list) else [eos]
        stopped = bool(ids and ids[-1] in eos_ids)
        token_limit = len(ids) >= settings["max_new_tokens"] and not stopped
        error = None
        try:
            hooks.finalize(ids)
        except CaptureError as exc:
            error = str(exc)
        resolved = {**self.resolved, "effective_generation": effective.to_dict(), "seed": settings["seed"],
            "direction_sha256": hooks.direction_sha256,
            "cuda_peak_memory": [{"device": index, "allocated_bytes": torch.cuda.max_memory_allocated(index),
                                  "reserved_bytes": torch.cuda.max_memory_reserved(index)} for index in cuda_devices]}
        return GenerationResult(raw, ids, "eos" if stopped else "length" if token_limit else "stop",
            token_limit, resolved, hooks.tensors, hooks.metadata, hooks.unavailable, capture_error=error)

    def close(self):
        self.model = None
        self.processor = None
        self.modules = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
