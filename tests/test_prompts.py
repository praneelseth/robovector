import json

import pytest
from pydantic import ValidationError

from robot_refusal.prompts import build_messages, serialize
from robot_refusal.records import AgentTrial


@pytest.mark.parametrize("condition,images,tools", [("C1", False, False), ("C2", False, True),
                                                    ("C3", True, True), ("C4", True, False)])
def test_four_conditions(trial_data, image_path, condition, images, tools):
    trial_data.update(condition=condition, image_paths=[image_path])
    trial_data["evaluator_metadata"] = {"secret_hazard_label": "DO_NOT_SHOW"}
    trial = AgentTrial.model_validate(trial_data)
    messages, treatment = build_messages(trial)
    contents = messages[1]["content"]
    assert [c["type"] for c in contents] == ["text", "text"] + (["image"] if images else []) + ["text"]
    assert contents[0]["text"].endswith(trial.scene_description)
    assert contents[-1]["text"].endswith(trial.instruction)
    assert "DO_NOT_SHOW" not in serialize(messages)
    assert "evaluator_metadata" not in serialize(messages)
    assert bool(treatment["tool_condition_addition"]) == tools
    assert bool(treatment["tool_catalog"]) == tools
    assert [item["image"] for item in contents if item["type"] == "image"] == ([image_path] if images else [])
    assert messages[0]["content"][0]["text"] == trial.prompts.system_instruction + treatment["tool_condition_addition"]


def test_matched_content_and_checkpoint(trial_data, image_path):
    trials = [AgentTrial.model_validate({**trial_data, "condition": condition, "image_paths": [image_path]})
              for condition in ("C1", "C2", "C3", "C4")]
    messages = [build_messages(trial)[0] for trial in trials]
    assert all(trial.model == trials[0].model for trial in trials)
    assert messages[0][0] == messages[3][0]
    assert messages[1][0] == messages[2][0]
    assert messages[0][1] == messages[1][1]
    assert messages[2][1] == messages[3][1]
    text = [[part for part in msg[1]["content"] if part["type"] == "text"] for msg in messages]
    assert all(parts == text[0] for parts in text)


@pytest.mark.parametrize("condition", ["C3", "C4"])
def test_image_required(trial_data, condition):
    with pytest.raises(ValidationError, match="requires at least one local image"):
        AgentTrial.model_validate({**trial_data, "condition": condition})


def test_missing_and_corrupt_inputs(trial_data, tmp_path):
    missing = AgentTrial.model_validate({**trial_data, "condition": "C3", "image_paths": [str(tmp_path / "absent.png")]})
    with pytest.raises(FileNotFoundError, match="Image file"):
        missing.check_input_files()
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")
    with pytest.raises(OSError):
        AgentTrial.model_validate({**trial_data, "image_paths": [str(invalid)]}).check_input_files()


def test_defaults_and_input_schema(trial_data):
    trial = AgentTrial.model_validate(trial_data)
    assert (trial.mode, trial.max_turns, trial.max_total_commands, trial.max_new_tokens_per_turn) == ("first_response", 1, 12, 1024)
    assert AgentTrial.model_validate({**trial_data, "mode": "bounded_queue"}).max_turns == 6
    assert trial == AgentTrial.model_validate_json(trial.model_dump_json())
    assert "condition" in AgentTrial.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        AgentTrial.model_validate({**trial_data, "max_total_commands": "12"})
    with pytest.raises(ValidationError):
        AgentTrial.model_validate({**trial_data, "max_turns": 7})
    with pytest.raises(ValidationError):
        AgentTrial.model_validate({**trial_data, "object_manifest": trial_data["object_manifest"] * 2})
    with pytest.raises(ValidationError):
        AgentTrial.model_validate({**trial_data, "model": {"model_id": "another/model"}})


def test_input_path_resolution(trial_data, tmp_path):
    trial_data["image_paths"] = ["scene.png"]
    trial_data["model"] = {"model_dir": "snapshot"}
    trial_data["intervention"] = {"layer": 0, "direction_path": "direction.pt"}
    trial = AgentTrial.model_validate(trial_data).resolve_paths(tmp_path)
    assert trial.image_paths == [str(tmp_path / "scene.png")]
    assert trial.model.model_dir == str(tmp_path / "snapshot")
    assert trial.intervention.direction_path == str(tmp_path / "direction.pt")
