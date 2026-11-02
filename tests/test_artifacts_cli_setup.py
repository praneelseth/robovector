import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from robot_refusal.agent import run_trial
from robot_refusal.artifacts import write_result
from robot_refusal.cli import main
from robot_refusal.model_setup import MANIFEST_NAME, read_snapshot, setup_model
from robot_refusal.models.fake import FakeAdapter
from robot_refusal.records import AgentTrial, AgentTrialResult, MODEL_ID


def test_json_roundtrip_and_no_overwrite(trial, tmp_path):
    result = run_trial(trial, FakeAdapter())
    path = write_result(result, tmp_path)
    restored = AgentTrialResult.model_validate_json(path.read_text())
    assert restored == result
    assert restored.activation_file is None
    assert not (tmp_path / "activations.pt").exists()
    with pytest.raises(FileExistsError):
        write_result(result, tmp_path)


def test_tensor_artifact_and_sidecar(trial_data, tmp_path):
    class TensorFixture(FakeAdapter):
        def generate(self, prepared, capture_config, intervention=None):
            # A synthetic artifact fixture, not model integration evidence.
            from robot_refusal.records import CaptureConfig
            output = super().generate(prepared, CaptureConfig())
            output.activations = {"layer_0.prompt_4.residual": torch.ones(3)}
            output.activation_metadata = [{"key": "layer_0.prompt_4.residual", "layer": 0,
                "module": "fixture.0", "token_id": 42, "sequence_position": 4, "stage": "prompt",
                "dtype": "torch.float32", "shape": [3]}]
            return output

    trial = AgentTrial.model_validate({**trial_data, "capture": {"enabled": True}})
    result = run_trial(trial, TensorFixture())
    write_result(result, tmp_path)
    tensors = torch.load(tmp_path / "activations.pt", weights_only=True)
    index = json.loads((tmp_path / "activations.json").read_text())
    assert index["trial_id"] == trial.trial_id
    assert index["backend"] == "fake"
    assert set(tensors) == {row["key"] for row in index["entries"]}
    assert torch.equal(tensors["turn_0.layer_0.prompt_4.residual"], torch.ones(3))
    assert "_activation_tensors" not in (tmp_path / "result.json").read_text()


def test_fake_capture_is_explicit_failure_with_artifacts(trial_data, tmp_path):
    trial = AgentTrial.model_validate({**trial_data, "capture": {"enabled": True}})
    result = run_trial(trial, FakeAdapter())
    assert result.exceptions
    write_result(result, tmp_path)
    assert torch.load(tmp_path / "activations.pt", weights_only=True) == {}
    assert json.loads((tmp_path / "activations.json").read_text())["exceptions"]


def test_cli_smoke_and_schema(tmp_path):
    example = Path(__file__).parents[1] / "examples/benign_trial.json"
    process = subprocess.run([sys.executable, "-m", "robot_refusal", "run-trial", "--trial", str(example),
                              "--backend", "fake", "--output", str(tmp_path / "run")], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["termination_reason"] == "first_response"
    assert result["command_queue"][0]["acknowledgment"]["executed"] is False
    schema = tmp_path / "schema.json"
    assert main(["schema", "--output", str(schema)]) == 0
    assert json.loads(schema.read_text())["title"] == "AgentTrial"


def test_cli_missing_image_and_overwrite(trial_data, tmp_path, capsys):
    trial_file = tmp_path / "trial.json"
    trial_file.write_text(json.dumps({**trial_data, "condition": "C3", "image_paths": ["missing.png"]}))
    args = ["run-trial", "--trial", str(trial_file), "--output", str(tmp_path / "out")]
    assert main(args) == 2
    assert "Image file does not exist" in capsys.readouterr().err
    trial_file.write_text(json.dumps(trial_data))
    assert main(args) == 0
    assert main(args) == 2
    assert "overwrite" in capsys.readouterr().err


def test_snapshot_setup_pins_commit(monkeypatch, tmp_path):
    import huggingface_hub

    commit = "a" * 40
    directory = tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct/snapshots" / commit
    directory.mkdir(parents=True)
    calls = []

    class API:
        def model_info(self, model_id, revision):
            assert model_id == MODEL_ID
            assert revision == "main"
            return type("Info", (), {"sha": commit})()

    def download(**kwargs):
        calls.append(kwargs)
        return str(directory)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    manifest = setup_model(MODEL_ID, str(tmp_path))
    assert calls == [{"repo_id": MODEL_ID, "revision": commit, "cache_dir": str(tmp_path)}]
    assert manifest["model_commit"] == commit
    assert read_snapshot(str(directory), MODEL_ID) == (directory, manifest)
    assert (directory / MANIFEST_NAME).is_file()


def test_unpinned_or_mismatched_snapshot_rejected(tmp_path):
    with pytest.raises(ValueError, match="setup-model"):
        read_snapshot(str(tmp_path), MODEL_ID)
    with pytest.raises(ValueError, match="--model-dir"):
        read_snapshot(None, MODEL_ID)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"model_id": MODEL_ID, "model_commit": "main"}))
    with pytest.raises(ValueError, match="commit is invalid"):
        read_snapshot(str(tmp_path), MODEL_ID)
    other = tmp_path / "snapshots" / ("b" * 40)
    other.mkdir(parents=True)
    (other / MANIFEST_NAME).write_text(json.dumps({"model_id": MODEL_ID, "model_commit": "a" * 40}))
    with pytest.raises(ValueError, match="differ"):
        read_snapshot(str(other), MODEL_ID)
