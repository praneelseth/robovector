"""Versioned, policy-neutral symbolic commands. There is no physical state."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import ValidationError

from .records import Acknowledgment, CallRecord, QueuedCommand, Record, SceneObject, TOOL_SCHEMA_VERSION


class PickArgs(Record):
    object_id: str


class PlaceArgs(PickArgs):
    target_id: str
    relation: Literal["on", "inside", "next_to"]


class MoveArgs(PickArgs):
    target_id: str
    target_part: str
    relation: Literal["above", "contact", "inside"]


class PourArgs(Record):
    source_id: str
    target_id: str


class FinishArgs(Record):
    pass


class ToolCall(Record):
    name: str
    arguments: dict[str, Any]


class Envelope(Record):
    assistant_text: str
    calls: list[Any]


ARGUMENTS = {
    "pick": PickArgs,
    "place": PlaceArgs,
    "move_held_object": MoveArgs,
    "pour": PourArgs,
    "release": PickArgs,
    "finish": FinishArgs,
}
DESCRIPTIONS = {
    "pick": "Request grasping the named object.",
    "place": "Request placing an object on, inside, or next to a target.",
    "move_held_object": "Request moving a held object relative to a visible target region.",
    "pour": "Request pouring from a named container into another.",
    "release": "Request releasing the named object.",
    "finish": "End command submission; must be the final call in a batch.",
}


def tool_catalog() -> dict[str, Any]:
    return {"version": TOOL_SCHEMA_VERSION, "tools": [
        {"name": name, "description": DESCRIPTIONS[name], "arguments": schema.model_json_schema()}
        for name, schema in ARGUMENTS.items()
    ]}


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def parse_response(raw: str, manifest: list[SceneObject], turn_index: int, remaining: int) -> dict[str, Any]:
    """Validate the entire batch before dispatch; never extract JSON from prose."""
    result = {"assistant_text": raw, "response_kind": "unstructured_response", "format_valid": False,
              "parsed_envelope": None, "calls": [], "validation_errors": [], "budget_exceeded": False}
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (ValueError, TypeError) as exc:
        result["validation_errors"] = [str(exc)]
        return result
    result["parsed_envelope"] = data
    result["response_kind"] = "invalid_envelope"
    raw_calls = data.get("calls", []) if isinstance(data, dict) else []
    # Even invalid envelopes retain every array element and its original index.
    if isinstance(raw_calls, list):
        result["calls"] = [CallRecord(turn_index=turn_index, call_index=i, raw=call)
                           for i, call in enumerate(raw_calls)]
    try:
        envelope = Envelope.model_validate(data)
    except ValidationError as exc:
        result["validation_errors"] = [str(exc)]
        for call in result["calls"]:
            call.errors.append("Envelope validation failed; call not dispatched")
        return result
    result["assistant_text"] = envelope.assistant_text
    objects = {obj.object_id: obj for obj in manifest}
    for record in result["calls"]:
        try:
            call = ToolCall.model_validate(record.raw)
            record.parsed = call.model_dump()
            if call.name not in ARGUMENTS:
                raise ValueError(f"Unknown tool: {call.name}")
            args = ARGUMENTS[call.name].model_validate(call.arguments).model_dump()
            for name, identifier in args.items():
                if name.endswith("_id") and identifier not in objects:
                    raise ValueError(f"Unknown {name}: {identifier}")
            if call.name == "move_held_object":
                part = args["target_part"]
                if part != "whole_object" and part not in objects[args["target_id"]].visible_parts:
                    raise ValueError(f"Unknown visible target_part: {part}")
            if call.name == "finish" and record.call_index != len(result["calls"]) - 1:
                raise ValueError("finish must be the final call")
            record.valid = True
        except (ValidationError, ValueError) as exc:
            record.errors.append(str(exc))
    # finish is a submission-control command and counts toward the command cap.
    if len(result["calls"]) > remaining:
        result["budget_exceeded"] = True
        result["validation_errors"].append(f"Batch exceeds remaining command budget ({remaining})")
        for call in result["calls"][remaining:]:
            call.valid = False
            call.errors.append("Command budget exceeded")
    result["format_valid"] = all(call.valid for call in result["calls"]) and not result["validation_errors"]
    result["response_kind"] = "tool_envelope" if result["format_valid"] else "invalid_calls"
    return result


class CommandQueue:
    def __init__(self, max_commands: int):
        self.max_commands = max_commands
        self.commands: list[QueuedCommand] = []

    @property
    def remaining(self) -> int:
        return self.max_commands - len(self.commands)

    def submit_batch(self, calls: list[CallRecord]) -> list[Acknowledgment]:
        if len(calls) > self.remaining or any(not c.valid or c.parsed is None for c in calls):
            raise ValueError("Only a fully validated batch within the budget can be queued")
        acknowledgments = []
        for call in calls:
            call_id = f"call_{len(self.commands) + 1:03d}"
            ack = Acknowledgment(call_id=call_id)
            self.commands.append(QueuedCommand(call_id=call_id, turn_index=call.turn_index,
                call_index=call.call_index, name=call.parsed["name"], arguments=call.parsed["arguments"],
                acknowledgment=ack))
            call.queued = True
            call.call_id = call_id
            acknowledgments.append(ack)
        return acknowledgments
