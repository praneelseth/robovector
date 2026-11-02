import json

import pytest

from robot_refusal.agent import run_trial
from robot_refusal.models.fake import FakeAdapter
from robot_refusal.records import AgentTrial


PICK = {"name": "pick", "arguments": {"object_id": "block_1"}}
FINISH = {"name": "finish", "arguments": {}}


def envelope(calls, text="Proposed commands."):
    return json.dumps({"assistant_text": text, "calls": calls})


def make_trial(data, **changes):
    return AgentTrial.model_validate({**data, **changes})


def test_primary_default_and_exact_evidence(trial_data):
    raw = envelope([PICK])
    trial = make_trial(trial_data, max_turns=6)
    result = run_trial(trial, FakeAdapter([raw]))
    assert len(result.turns) == 1
    assert result.termination_reason == "first_response"
    turn = result.turns[0]
    assert turn.raw_output == raw
    assert turn.generated_token_ids == list(raw.encode())
    assert turn.calls[0].raw == PICK
    assert turn.calls[0].parsed == PICK
    assert turn.calls[0].queued
    assert turn.calls[0].call_id == "call_001"
    assert json.loads(turn.serialized_messages) == turn.messages
    assert result.observations["assistant_refusal_language"] is None
    assert result.observations["proposed_commands"] is None
    assert result.environment["packages"]["torch"]


@pytest.mark.parametrize("condition", ["C1", "C4"])
def test_conversation_does_not_dispatch_json(trial_data, image_path, condition):
    result = run_trial(make_trial(trial_data, condition=condition, image_paths=[image_path]), FakeAdapter([envelope([PICK])]))
    assert result.termination_reason == "conversation_response"
    assert result.command_queue == []
    assert result.turns[0].format_valid is None


def test_plain_refusal_not_safety_labeled(trial):
    result = run_trial(trial, FakeAdapter(["I cannot help with that request."]))
    assert result.termination_reason == "invalid_output"
    assert result.turns[0].response_kind == "unstructured_response"
    assert result.turns[0].assistant_text == "I cannot help with that request."
    assert result.turns[0].format_valid is False
    assert result.observations["assistant_refusal_language"] is None
    assert "unsafe" not in result.model_dump_json()
    assert result.command_queue == []


def test_refusal_and_commands_both_preserved(trial):
    raw = envelope([PICK], "I refuse this request.")
    result = run_trial(trial, FakeAdapter([raw]))
    assert result.turns[0].assistant_text == "I refuse this request."
    assert len(result.command_queue) == 1
    assert result.turns[0].calls[0].raw == PICK


def test_invalid_batch_no_partial_dispatch(trial):
    invalid = {"name": "pick", "arguments": {"object_id": "unknown"}}
    raw = envelope([PICK, invalid, FINISH])
    result = run_trial(trial, FakeAdapter([raw]))
    assert result.termination_reason == "invalid_output"
    assert [call.raw for call in result.turns[0].calls] == [PICK, invalid, FINISH]
    assert [call.valid for call in result.turns[0].calls] == [True, False, True]
    assert result.command_queue == []


@pytest.mark.parametrize("responses,changes,reason,count", [
    ([envelope([FINISH])], {}, "finish", 1),
    ([envelope([])], {}, "no_calls", 0),
    ([envelope([PICK])], {"max_total_commands": 1}, "command_budget", 1),
    ([envelope([PICK, PICK])], {"max_total_commands": 1}, "command_budget", 0),
    ([envelope([PICK])], {"max_turns": 1}, "turn_budget", 1),
    ([envelope([PICK]), envelope([PICK])], {"max_turns": 2}, "turn_budget", 2),
    ([envelope([PICK]), envelope([PICK, PICK])], {"max_total_commands": 2}, "command_budget", 1),
])
def test_continuation_termination(trial_data, responses, changes, reason, count):
    result = run_trial(make_trial(trial_data, mode="bounded_queue", **changes), FakeAdapter(responses))
    assert result.termination_reason == reason
    assert len(result.command_queue) == count
    assert result.incomplete == reason.endswith("budget")


def test_continuation_feedback_and_original_image(trial_data, image_path):
    trial = make_trial(trial_data, condition="C3", image_paths=[image_path], mode="bounded_queue")
    result = run_trial(trial, FakeAdapter([envelope([PICK]), envelope([FINISH])]))
    assert result.termination_reason == "finish"
    assert len(result.turns) == 2
    second = result.turns[1].messages
    assert second[:2] == result.turns[0].messages
    assert [part for message in second for part in message["content"] if part["type"] == "image"] == [{"type": "image", "image": image_path}]
    feedback = second[-1]["content"][0]["text"]
    assert '"executed":false' in feedback
    assert '"observation_updated":false' in feedback
    assert "scene has not been observed again" in feedback
    assert "original observation" in feedback
    assert result.command_queue[-1].call_id == "call_002"


def test_trials_are_isolated_even_with_adapter_reuse(trial_data):
    adapter = FakeAdapter([envelope([PICK])])
    first = run_trial(make_trial(trial_data), adapter)
    second = run_trial(make_trial(trial_data, trial_id="second"), adapter)
    assert adapter.closed
    assert len(second.turns[0].messages) == 2
    assert len(second.command_queue) == 1
    assert second.command_queue[0].call_id == "call_001"
    first.turns[0].messages[0]["role"] = "changed"
    assert second.turns[0].messages[0]["role"] == "system"


def test_truncation_is_incomplete_and_raw_retained(trial_data):
    result = run_trial(make_trial(trial_data, max_new_tokens_per_turn=5), FakeAdapter([envelope([PICK])]))
    assert result.token_limit_reached and result.incomplete
    assert result.turns[0].raw_output == '{"ass'
    assert result.turns[0].generation_finish_reason == "length"
    assert result.termination_reason == "invalid_output"
    assert result.command_queue == []


def test_primary_at_command_cap_marked_incomplete(trial_data):
    result = run_trial(make_trial(trial_data, max_total_commands=1), FakeAdapter([envelope([PICK])]))
    assert result.incomplete


def test_exceptions_are_explicit_and_adapter_closed(trial):
    adapter = FakeAdapter([])
    result = run_trial(trial, adapter)
    assert result.termination_reason == "generation_error"
    assert result.exceptions[0]["type"] == "ValueError"
    assert result.turns[0].exception
    assert result.turns[0].prepared["prompt_token_ids"]
    assert adapter.closed


def test_load_failure_record(trial):
    class FailingAdapter(FakeAdapter):
        def load(self, config):
            raise RuntimeError("unavailable")

    adapter = FailingAdapter()
    result = run_trial(trial, adapter)
    assert result.termination_reason == "load_error"
    assert result.exceptions[0]["message"] == "unavailable"
    assert adapter.closed


def test_capture_failure_preserves_generated_evidence(trial):
    class CaptureFailure(FakeAdapter):
        def generate(self, *args):
            generated = super().generate(*args)
            generated.capture_error = "Missing layer"
            return generated

    result = run_trial(trial, CaptureFailure([envelope([PICK])]))
    assert result.termination_reason == "capture_error"
    assert result.turns[0].raw_output == envelope([PICK])
    assert result.turns[0].generated_token_ids
    assert result.command_queue == []
