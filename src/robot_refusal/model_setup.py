"""Explicit snapshot download is the only network-enabled model operation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .records import MODEL_ID

MANIFEST_NAME = "robot_refusal_snapshot.json"


def setup_model(model_id: str, cache_dir: str, revision: str = "main") -> dict:
    if model_id != MODEL_ID:
        raise ValueError(f"Only {MODEL_ID} is supported")
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().model_info(model_id, revision=revision).sha
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The Hub did not resolve an exact model commit")
    cache = Path(cache_dir).expanduser().resolve()
    snapshot = Path(snapshot_download(repo_id=model_id, revision=commit, cache_dir=str(cache)))
    manifest = {"model_id": model_id, "requested_revision": revision, "model_commit": commit,
                "snapshot_path": str(snapshot.resolve()), "cache_dir": str(cache),
                "resolved_at_utc": datetime.now(timezone.utc).isoformat()}
    (snapshot / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_snapshot(model_dir: str | None, model_id: str) -> tuple[Path, dict]:
    if model_dir is None:
        raise ValueError("Qwen requires --model-dir pointing to a snapshot from setup-model")
    path = Path(model_dir).expanduser().resolve()
    if not (path / MANIFEST_NAME).is_file():
        raise ValueError(f"Missing {MANIFEST_NAME} in {path}; run setup-model explicitly first")
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    if manifest.get("model_id") != model_id or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("model_commit", "")):
        raise ValueError("Snapshot manifest model ID or exact commit is invalid")
    if path.parent.name == "snapshots" and path.name != manifest["model_commit"]:
        raise ValueError("Snapshot directory and recorded model commit differ")
    return path, manifest
