"""One-trial orchestration; no dataset selection or behavioral labels."""

from copy import deepcopy
from importlib import metadata
import platform
import random
import time

from .models.base import ModelAdapter
from .prompts import build_messages, continuation_message, serialize
from .records import AgentTrial, AgentTrialResult, TurnRecord
from .tools import CommandQueue, parse_response


def environment_record() -> dict:
    import torch

    packages = {}
    for name in ("robot-refusal", "torch", "torchvision", "transformers", "Pillow", "pydantic", "huggingface-hub",
                 "accelerate", "safetensors", "tokenizers", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                      "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())],
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def generation_settings(trial: AgentTrial) -> dict:
    return {**trial.generation.model_dump(), "max_new_tokens": trial.max_new_tokens_per_turn,
            "seed": trial.seed, "num_beams": 1, "num_return_sequences": 1, "use_cache": True}


def run_trial(trial: AgentTrial, adapter: ModelAdapter) -> AgentTrialResult:
    import numpy as np
    import torch

    trial.check_input_files()
    started = time.perf_counter()
    random.seed(trial.seed)
    np.random.seed(trial.seed)
    torch.manual_seed(trial.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial.seed)
    messages, treatment = build_messages(trial)
    queue = CommandQueue(trial.max_total_commands)
    result = AgentTrialResult(trial_id=trial.trial_id, task_id=trial.task_id, pair_id=trial.pair_id,
        condition=trial.condition, backend=adapter.backend, resolved_configuration=trial.model_dump(),
        environment=environment_record(), prompt_treatment=treatment)
    turn = None
    try:
        adapter.load(trial.model)
        limit = 1 if trial.mode == "first_response" else trial.max_turns
        for index in range(limit):
            turn_started = time.perf_counter()
            turn = TurnRecord(turn_index=index, messages=deepcopy(messages), serialized_messages=serialize(messages))
            result.turns.append(turn)
            try:
                prepared = adapter.prepare(messages, generation_settings(trial))
                turn.prepared = deepcopy(prepared.metadata)
                generated = adapter.generate(prepared, trial.capture, trial.intervention)
                # Preserve generation evidence before interpreting any protocol content.
                turn.raw_output = generated.raw_text
                turn.generated_token_ids = generated.generated_ids
                turn.generation_finish_reason = generated.finish_reason
                turn.token_limit_reached = generated.token_limit_reached
                result.token_limit_reached |= generated.token_limit_reached
                turn.resolved_model_config = generated.resolved_configuration
                turn.next_token_statistics = generated.next_token_statistics
                for key, tensor in generated.activations.items():
                    full_key = f"turn_{index}.{key}"
                    result._activation_tensors[full_key] = tensor.detach().cpu().clone()
                    turn.activation_keys.append(full_key)
                for row in generated.activation_metadata:
                    turn.activation_metadata.append({**row, "key": f"turn_{index}.{row['key']}", "turn_index": index})
                turn.unavailable_activations = generated.unavailable_activations
                if generated.capture_error:
                    from .activations import CaptureError
                    raise CaptureError(generated.capture_error)
                if not trial.uses_tools:
                    turn.assistant_text = generated.raw_text
                    turn.response_kind = "conversation_response"
                    result.termination_reason = "conversation_response"
                    result.incomplete = generated.token_limit_reached
                    break
                parsed = parse_response(generated.raw_text, trial.object_manifest, index, queue.remaining)
                for field in ("assistant_text", "response_kind", "format_valid", "parsed_envelope", "calls", "validation_errors"):
                    setattr(turn, field, parsed[field])
                if not turn.format_valid:
                    result.termination_reason = "command_budget" if parsed["budget_exceeded"] else "invalid_output"
                    result.incomplete = True
                    break
                turn.acknowledgments = queue.submit_batch(turn.calls)
                if generated.token_limit_reached:
                    result.termination_reason = "token_limit"
                    result.incomplete = True
                    break
                if trial.mode == "first_response":
                    result.termination_reason = "first_response"
                    result.incomplete = queue.remaining == 0 and not any(
                        call.parsed["name"] == "finish" for call in turn.calls)
                    break
                if any(call.parsed["name"] == "finish" for call in turn.calls):
                    result.termination_reason = "finish"
                    break
                if not turn.calls:
                    result.termination_reason = "no_calls"
                    break
                if queue.remaining == 0:
                    result.termination_reason = "command_budget"
                    result.incomplete = True
                    break
                if index + 1 == limit:
                    result.termination_reason = "turn_budget"
                    result.incomplete = True
                    break
                messages.append({"role": "assistant", "content": [{"type": "text", "text": generated.raw_text}]})
                messages.append(continuation_message(trial, turn.acknowledgments))
            finally:
                turn.latency_seconds = time.perf_counter() - turn_started
    except Exception as exc:
        from .activations import CaptureError
        error = {"type": type(exc).__name__, "message": str(exc)}
        result.exceptions.append({**error, "turn_index": turn.turn_index if turn else None})
        if turn:
            turn.exception = error
        result.termination_reason = "capture_error" if isinstance(exc, CaptureError) else (
            "generation_error" if turn else "load_error")
        result.incomplete = True
    finally:
        try:
            adapter.close()
        except Exception as exc:
            result.exceptions.append({"type": type(exc).__name__, "message": str(exc), "stage": "close"})
            result.termination_reason = "close_error"
            result.incomplete = True
        result.command_queue = queue.commands
        result.latency_seconds = time.perf_counter() - started
    return result
