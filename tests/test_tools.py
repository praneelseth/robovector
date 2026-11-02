import json

import pytest

from robot_refusal.tools import CommandQueue, parse_response, tool_catalog


def parse(calls, trial, **kwargs):
    return parse_response(json.dumps({"assistant_text": "", "calls": calls}), trial.object_manifest, 0, kwargs.get("remaining", 12))


VALID_CALLS = [
    {"name": "pick", "arguments": {"object_id": "block_1"}},
    {"name": "place", "arguments": {"object_id": "block_1", "target_id": "bowl_1", "relation": "inside"}},
    {"name": "move_held_object", "arguments": {"object_id": "block_1", "target_id": "bowl_1", "target_part": "rim", "relation": "contact"}},
    {"name": "pour", "arguments": {"source_id": "block_1", "target_id": "bowl_1"}},
    {"name": "release", "arguments": {"object_id": "block_1"}},
    {"name": "finish", "arguments": {}},
]


def test_all_signatures_and_acknowledgments(trial):
    parsed = parse(VALID_CALLS, trial)
    assert parsed["format_valid"]
    queue = CommandQueue(12)
    acknowledgments = queue.submit_batch(parsed["calls"])
    assert len(queue.commands) == 6
    for i, ack in enumerate(acknowledgments):
        assert ack.model_dump() == {"call_id": f"call_{i+1:03d}", "status": "queued", "executed": False, "observation_updated": False}
    assert [call.raw for call in parsed["calls"]] == VALID_CALLS


@pytest.mark.parametrize("call", [
    {"name": "shell", "arguments": {"command": "example"}},
    {"name": "pick", "arguments": {}},
    {"name": "pick", "arguments": {"object_id": "block_1", "extra": True}},
    {"name": "pick", "arguments": {"object_id": "absent"}},
    {"name": "pick", "arguments": {"object_id": 1}},
    {"name": "pick", "arguments": None},
    {"name": "pick", "arguments": {"object_id": "block_1"}, "extra": 1},
    {"name": "place", "arguments": {"object_id": "block_1", "target_id": "bowl_1", "relation": "under"}},
    {"name": "move_held_object", "arguments": {"object_id": "block_1", "target_id": "bowl_1", "target_part": "hidden", "relation": "contact"}},
    {"name": "move_held_object", "arguments": {"object_id": "block_1", "target_id": "bowl_1", "relation": "contact"}},
    {"name": "finish", "arguments": {"object_id": "block_1"}},
    "not an object", None, 12,
])
def test_invalid_calls_retained(trial, call):
    parsed = parse([call], trial)
    assert not parsed["format_valid"]
    assert parsed["calls"][0].raw == call
    assert not parsed["calls"][0].valid
    assert parsed["calls"][0].errors
    with pytest.raises(ValueError):
        CommandQueue(12).submit_batch(parsed["calls"])


@pytest.mark.parametrize("calls", [[VALID_CALLS[-1], VALID_CALLS[0]], [VALID_CALLS[-1], VALID_CALLS[-1]]])
def test_finish_must_be_last(trial, calls):
    parsed = parse(calls, trial)
    assert not parsed["format_valid"]
    assert "final" in parsed["calls"][0].errors[0]


def test_budget_is_atomic_and_preserves_full_batch(trial):
    parsed = parse(VALID_CALLS, trial, remaining=2)
    assert parsed["budget_exceeded"]
    assert not parsed["format_valid"]
    assert [call.raw for call in parsed["calls"]] == VALID_CALLS
    assert [call.valid for call in parsed["calls"]] == [True, True, False, False, False, False]
    queue = CommandQueue(2)
    with pytest.raises(ValueError):
        queue.submit_batch(parsed["calls"])
    assert queue.commands == []


@pytest.mark.parametrize("raw", ["I cannot help with that.", "```json\n{\"assistant_text\":\"\",\"calls\":[]}\n```",
                                 'Here is a call: {"name":"pick","arguments":{"object_id":"block_1"}}',
                                 '{"assistant_text":', '{"assistant_text":"","calls":[],"calls":[]}',
                                 '{"assistant_text":"","calls":[NaN]}'])
def test_no_prose_extraction_or_json_repair(trial, raw):
    result = parse_response(raw, trial.object_manifest, 0, 12)
    assert result["response_kind"] == "unstructured_response"
    assert not result["format_valid"]
    assert result["assistant_text"] == raw
    assert result["calls"] == []


@pytest.mark.parametrize("data", [[], None, 123, {"assistant_text": ""}, {"assistant_text": 42, "calls": VALID_CALLS},
                                  {"assistant_text": "", "calls": {}}, {"assistant_text": "", "calls": VALID_CALLS, "extra": 1}])
def test_invalid_envelope_keeps_raw_data(trial, data):
    parsed = parse_response(json.dumps(data), trial.object_manifest, 2, 12)
    assert not parsed["format_valid"]
    assert parsed["parsed_envelope"] == data
    if isinstance(data, dict) and isinstance(data.get("calls"), list):
        assert [call.raw for call in parsed["calls"]] == data["calls"]
        assert all(call.turn_index == 2 for call in parsed["calls"])


def test_no_physical_or_hazard_state(trial_data):
    from robot_refusal.records import AgentTrial
    trial_data["object_manifest"] += [{"object_id": "knife_1", "name": "knife", "visible_parts": []}]
    trial = AgentTrial.model_validate(trial_data)
    # Release without a preceding grasp, contact with a region, and pour from an
    # object not described as a container all remain proposals for later scoring.
    calls = [VALID_CALLS[4], {"name": "move_held_object", "arguments": {
        "object_id": "knife_1", "target_id": "bowl_1", "target_part": "whole_object", "relation": "contact"}}, VALID_CALLS[3]]
    assert parse(calls, trial)["format_valid"]


def test_versioned_catalog():
    catalog = tool_catalog()
    assert catalog["version"] == "symbolic-v1"
    assert [entry["name"] for entry in catalog["tools"]] == [call["name"] for call in VALID_CALLS]
    place = catalog["tools"][1]["arguments"]
    assert place["additionalProperties"] is False
    assert place["properties"]["relation"]["enum"] == ["on", "inside", "next_to"]
