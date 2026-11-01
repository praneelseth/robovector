import json
from pathlib import Path

import pytest

from robot_refusal.records import AgentTrial


@pytest.fixture
def trial_data():
    return json.loads((Path(__file__).parents[1] / "examples/benign_trial.json").read_text())


@pytest.fixture
def trial(trial_data):
    return AgentTrial.model_validate(trial_data)


@pytest.fixture
def image_path(tmp_path):
    from PIL import Image

    path = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), "white").save(path)
    return str(path)
