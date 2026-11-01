"""Matched condition builders. Evaluator metadata never enters the messages."""

import json
from typing import Any

from .records import AgentTrial
from .tools import tool_catalog


def serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_messages(trial: AgentTrial) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = tool_catalog() if trial.uses_tools else None
    addition = ("\n\n" + trial.prompts.tool_instruction + "\n" + serialize(catalog)) if catalog else ""
    system = trial.prompts.system_instruction + addition
    content = [
        {"type": "text", "text": trial.prompts.scene_label + "\n" + trial.scene_description},
        {"type": "text", "text": trial.prompts.manifest_label + "\n" +
            serialize([obj.model_dump() for obj in trial.object_manifest])},
    ]
    if trial.uses_images:
        content.extend({"type": "image", "image": path} for path in trial.image_paths)
    content.append({"type": "text", "text": trial.prompts.request_label + "\n" + trial.instruction})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    return messages, {"common_system_instruction": trial.prompts.system_instruction,
                      "tool_condition_addition": addition, "tool_catalog": catalog,
                      "image_paths_supplied": trial.image_paths if trial.uses_images else [],
                      "component_order": ["scene_description", "object_manifest", "images", "request"]}


def continuation_message(trial: AgentTrial, acknowledgments) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text":
        "Queue acknowledgments:\n" + serialize([ack.model_dump() for ack in acknowledgments]) +
        "\n" + trial.prompts.continuation_instruction}]}
