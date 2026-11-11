"""Bounded, idempotent GPU smoke harness. No robot execution, no arbitrary inputs."""
import base64
import io
import json
import os
from pathlib import Path
import threading
import time
import zipfile


class Model:
    def __init__(self, **kwargs):
        self.data_dir = Path(kwargs['data_dir'])
        self.lock = threading.Lock()
        self.cached_result = None

    def load(self):
        import torch
        from robot_refusal.models.qwen_vl import QwenVLAdapter
        from robot_refusal.records import ModelConfig
        if not torch.cuda.is_available():
            raise RuntimeError('This smoke deployment requires CUDA')
        self.root = Path('/tmp/qwen-smoke-work')
        self.root.mkdir(exist_ok=True)
        snapshot = self.root / 'snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in Path('/models/qwen').iterdir():
            target = snapshot / source.name
            if source.is_file() and not target.exists():
                target.symlink_to(source)
        metadata = json.loads((self.data_dir / 'checkpoint.json').read_text())
        (snapshot / 'robot_refusal_snapshot.json').write_text(json.dumps(dict(
            model_id='Qwen/Qwen3-VL-8B-Instruct', model_commit=metadata['sha'],
            snapshot_path=str(snapshot), requested_revision=metadata['sha'],
            cache_dir='/models/qwen', provenance='Baseten pinned HF weights mount')))
        self.model_config = dict(model_id='Qwen/Qwen3-VL-8B-Instruct', model_dir=str(snapshot),
                                device='cuda:0', dtype='bfloat16', attn_implementation='sdpa')
        self.adapter = QwenVLAdapter()
        self.adapter.load(ModelConfig.model_validate(self.model_config))
        self.model_loaded_at = time.time()

    def predict(self, request):
        if request != {'operation': 'smoke-v1'}:
            raise ValueError('Only the fixed smoke-v1 operation is supported')
        with self.lock:
            if self.cached_result is not None:
                return self.cached_result  # A repeated HTTP request never generates again.
            self.cached_result = self._run()
            return self.cached_result

    def _run(self):
        import torch
        from src.eval.check_capture import compare_capture
        from src.eval.integrated import run_integrated
        from robot_refusal.records import ModelConfig
        from transformers import StoppingCriteria, StoppingCriteriaList
        started = time.monotonic()
        deadline = started + 480
        output = self.root / 'artifacts'
        output.mkdir(exist_ok=True)
        owner = self

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Eight-minute inference budget reached')
                return False

        original_generate = self.adapter.model.generate
        def bounded_generate(*args, **kwargs):
            kwargs['stopping_criteria'] = StoppingCriteriaList([Deadline()])
            return original_generate(*args, **kwargs)
        self.adapter.model.generate = bounded_generate

        class SessionAdapter:
            backend = 'qwen'

            def load(self, config):
                if config != ModelConfig.model_validate(owner.model_config):
                    raise ValueError('Model configuration changed within smoke session')

            def prepare(self, *args, **kwargs):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Smoke deadline reached')
                return owner.adapter.prepare(*args, **kwargs)

            def generate(self, *args, **kwargs):
                return owner.adapter.generate(*args, **kwargs)

            def close(self):
                pass  # The deployment owns the shared model, not individual trials.

        error = None
        try:
            for vision in (False, True):
                content = ([{'type': 'image', 'image': str(self.data_dir / 'images/burner_real.png')}]
                           if vision else [])
                content.append({'type': 'text', 'text': 'Describe the scene briefly.'})
                prepared = self.adapter.prepare([{'role': 'user', 'content': content}], {})
                inputs = {k: v.to('cuda:0') for k, v in prepared.inputs.items()}
                eos = self.adapter.model.generation_config.eos_token_id
                report = compare_capture(self.adapter.model, inputs, self.adapter.modules,
                    self.adapter.model.model.language_model.norm,
                    generate=lambda: self.adapter.model.generate(**inputs, max_new_tokens=2,
                        do_sample=False, use_cache=True,
                        pad_token_id=eos[0] if isinstance(eos, list) else eos))
                report.update(model=self.model_config['model_id'], revision=self.adapter.snapshot['model_commit'],
                              vision=vision, prepared=prepared.metadata)
                (output / ('capture-vision.json' if vision else 'capture-text.json')).write_text(json.dumps(report, indent=2))
                if not report['allclose']:
                    raise RuntimeError('Capture equality failed; evaluation not started')
            cfg = dict(tasks=str(self.data_dir / 'tasks.jsonl'), task_ids=['burner', 'burner_safe'],
                conditions=['text_chat', 'text_tools', 'vision'], n_repeats=2,
                temperature=0.7, seed=42, max_new_tokens=256, truncation_max_new_tokens=1024,
                model_config=self.model_config, image_dir=str(self.data_dir / 'images'),
                run_dir=str(output / 'run'), output_dir=str(output))
            run_integrated(cfg, adapter_factory=SessionAdapter, adapter_factory_is_test=False)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            self.adapter.model.generate = original_generate
        status = dict(error=error, elapsed_seconds=time.monotonic()-started,
                      model=self.model_config['model_id'], revision=self.adapter.snapshot['model_commit'],
                      gpu=torch.cuda.get_device_name(0), budget_seconds=480)
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
        return dict(status=status, archive_base64=base64.b64encode(buffer.getvalue()).decode())
