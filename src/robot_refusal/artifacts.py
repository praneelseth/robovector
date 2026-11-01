"""JSON record and CPU-only tensor artifacts with an explicit index sidecar."""

import json
from pathlib import Path

from .records import AgentTrialResult


def write_result(result: AgentTrialResult, output: str | Path) -> Path:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in ("result.json", "activations.pt", "activations.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Refusing to overwrite trial artifacts in {directory}")
    configuration = result.resolved_configuration
    if configuration["capture"]["enabled"] or configuration.get("intervention"):
        import torch

        result.activation_file = "activations.pt"
        result.activation_sidecar = "activations.json"
        torch.save(result._activation_tensors, directory / result.activation_file)
        index = {"trial_id": result.trial_id, "backend": result.backend,
                 "model_commits": [turn.resolved_model_config.get("model_commit") for turn in result.turns],
                 "termination_reason": result.termination_reason,
                 "entries": [row for turn in result.turns for row in turn.activation_metadata],
                 "unavailable": [{"turn_index": turn.turn_index, **row} for turn in result.turns
                                 for row in turn.unavailable_activations],
                 "exceptions": result.exceptions}
        (directory / result.activation_sidecar).write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    destination = directory / "result.json"
    destination.write_text(result.model_dump_json(indent=2) + "\n")
    return destination
