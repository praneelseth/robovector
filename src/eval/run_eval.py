"""Single-generation evaluation driver; partner adapters own multi-turn behavior."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from data.prompt_templates import CONDITIONS, TEMPLATE_VERSION, build_messages
from src.eval.labeling import LABELER_VERSION, LABELS, REFUSAL_LABELS, classify_outcome
from src.interfaces import ModelWrapper, RobotTools


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def read_jsonl(path, *, recover_trailing=False):
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if not recover_trailing or i != len(lines) - 1:
                raise ValueError(f"Corrupt JSONL at {path}:{i+1}")
            # A killed writer can leave one incomplete trailing record.
            write_jsonl(path, records)
    return records


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(path)


def summarize(records, path):
    """Report explicit safety/capability/unclear rates over all valid trials."""
    groups = {}
    outcome_columns = ["safety_refusal", "capability_refusal", "unclear_refusal", "comply", "partial", "said_no_but_acted", "invalid_action", "error", "pending_truncated"]
    for record in records:
        if record.get('labeler_version', LABELER_VERSION) != LABELER_VERSION:
            raise ValueError('Legacy labeler version; explicitly relabel before summarizing')
        pending = record["outcome"] is None and record.get("truncated", False)
        if record["outcome"] not in LABELS and not pending:
            raise ValueError("Legacy/unknown outcome; explicitly relabel before summarizing")
        key = (record["model"], record["condition"], record["harm_label"])
        counts = groups.setdefault(key, {label: 0 for label in outcome_columns})
        counts["pending_truncated" if pending else record["outcome"]] += 1
    rate_columns = ["safety_refusal_rate", "capability_refusal_rate", "unclear_refusal_rate", "any_refusal_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "harm_label", "n_total", "n_valid", *outcome_columns, *rate_columns])
        writer.writeheader()
        for (model, condition, harm_label), counts in sorted(groups.items()):
            total = sum(counts.values())
            valid = total - counts["error"] - counts["pending_truncated"]
            rates = {label + "_rate": counts[label] / valid if valid else "" for label in REFUSAL_LABELS}
            rates["any_refusal_rate"] = sum(counts[label] for label in REFUSAL_LABELS) / valid if valid else ""
            writer.writerow(dict(model=model, condition=condition, harm_label=harm_label,
                                 n_total=total, n_valid=valid, **counts, **rates))


def run_eval(config: dict, *, model_factory=None, robot_tools: RobotTools | None = None,
             judge=None) -> Path:
    """Resume the specified run directory, rejecting changed experimental inputs.

    Injection: model_factory(name)->ModelWrapper, robot_tools.schemas(), and an
    optional judge(record)->label. Judge is off unless explicitly supplied/configured.
    A single writer per run directory is supported. Error records are completed
    outputs too; use a new run to retry them. Missing/corrupt activations are repaired.
    """
    cfg = dict(config)
    if cfg.get("engine") == "integrated":
        from src.eval.integrated import run_integrated
        return run_integrated(cfg)
    defaults = dict(models=["stub"], conditions=list(CONDITIONS), n_repeats=1,
                    temperature=0.0, seed=0, max_new_tokens=128, tasks="data/tasks.jsonl",
                    image_dir="data/images", image_source="real", output_dir="outputs",
                    include_drafts=False, model_factory="src.stubs.model:create_model",
                    tools_factory="src.stubs.robot_tools:RobotTools", judge=None,
                    action_tool_names=["move_object"])
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    if (model_factory is None and not cfg['model_factory'].startswith('src.stubs.')
            and any(c.endswith('tools') for c in cfg['conditions'])):
        raise ValueError('Real tool experiments must use engine: integrated and the shared JSON protocol')
    if cfg["n_repeats"] < 1 or cfg["max_new_tokens"] < 1 or cfg["temperature"] < 0:
        raise ValueError("Invalid generation settings")
    if not cfg["models"] or not cfg["conditions"] or not set(cfg["conditions"]) <= set(CONDITIONS):
        raise ValueError("Models and valid conditions are required")
    if len(set(cfg["models"])) != len(cfg["models"]) or len(set(cfg["conditions"])) != len(cfg["conditions"]):
        raise ValueError("Duplicate models or conditions")
    task_path = Path(cfg["tasks"])
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    tasks = read_jsonl(task_path)
    if cfg.get("task_ids"):
        missing = set(cfg["task_ids"]) - {t["task_id"] for t in tasks}
        if missing:
            raise ValueError(f"Unknown task IDs: {missing}")
        tasks = [t for t in tasks if t["task_id"] in cfg["task_ids"]]
    allowed_statuses = {"approved", "draft"} if cfg["include_drafts"] else {"approved"}
    tasks = [t for t in tasks
             if t.get("review_status", "approved") in allowed_statuses
             and not t.get("excluded_from_v1", False)]
    if not tasks:
        raise ValueError("No eligible tasks selected; rejected, deferred and v1-excluded tasks cannot run, even with include_drafts")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    for task in tasks:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", task["task_id"]):
            raise ValueError("Unsafe task ID")
        if task["harm_label"] not in ("harmful", "harmless"):
            raise ValueError("Invalid harm label")
    robot_tools = robot_tools or resolve(cfg["tools_factory"])()
    schemas = robot_tools.schemas()
    action_names = set(cfg["action_tool_names"])
    schema_names = {s["function"]["name"] for s in schemas}
    if any("tools" in c for c in cfg["conditions"]) and not action_names <= schema_names:
        raise ValueError("action_tool_names must be present in schemas")
    model_factory = model_factory or resolve(cfg["model_factory"])
    judge = judge or (resolve(cfg["judge"]) if cfg["judge"] else None)
    image_hashes = {}
    if any(c.startswith("vision") for c in cfg["conditions"]):
        for task in tasks:
            for key in task["image_ids"]:
                if key.endswith("_" + cfg["image_source"]):
                    p = Path(cfg["image_dir"]) / f"{key}.png"
                    image_hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    manifest = {"config": {k: v for k, v in cfg.items() if k != "run_dir"},
                "tasks": tasks, "image_sha256": image_hashes, "tool_schemas": schemas,
                "template_version": TEMPLATE_VERSION, "labeler_version": LABELER_VERSION,
                "judge_enabled": judge is not None}
    fingerprint = digest(manifest)
    output = Path(cfg["output_dir"])
    run_dir = Path(cfg["run_dir"]) if cfg.get("run_dir") else output / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration/data differ; use a new run directory")
    else:
        manifest_path.write_text(json.dumps(dict(fingerprint=fingerprint, **manifest), indent=2), encoding="utf-8")
    activation_namespace = run_dir.name + "-" + digest(str(run_dir.resolve()))[:8]
    result_path = run_dir / "results.jsonl"
    records = read_jsonl(result_path, recover_trailing=True)
    by_key = {r["run_key"]: r for r in records}

    def persist():
        write_jsonl(result_path, list(by_key.values()))
        write_jsonl(run_dir / "activations.jsonl", [
            {k: r[k] for k in ("run_key", "model", "task_id", "pair_id", "harm_label", "condition", "repeat", "seed", "activation_path", "activation_shape", "activation_sha256")}
            for r in by_key.values() if r.get("activation_path")])
        summarize(list(by_key.values()), run_dir / "summary.csv")

    for model_name in cfg["models"]:
        model: ModelWrapper | None = None
        for task, condition, repeat in itertools.product(tasks, cfg["conditions"], range(cfg["n_repeats"])):
            identity = [fingerprint, model_name, task["task_id"], condition, repeat]
            key = digest(identity)
            old = by_key.get(key)
            if old:
                p = Path(old["activation_path"]) if old.get("activation_path") else None
                if old["outcome"] == "error" or (p and p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == old["activation_sha256"]):
                    continue
            trial_seed = (cfg["seed"] + int(digest([model_name, task["task_id"], condition, repeat])[:8], 16)) % (2**32)
            record = dict(run_key=key, fingerprint=fingerprint, model=model_name,
                          task_id=task["task_id"], pair_id=task["pair_id"], harm_label=task["harm_label"],
                          twin_type=task.get("twin_type"), condition=condition, repeat=repeat,
                          seed=trial_seed, temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"],
                          image_source=cfg["image_source"], text="", tool_calls=[], messages=[],
                          activation_path=None, activation_shape=None, activation_sha256=None,
                          outcome="error", error=None, labeler_version=LABELER_VERSION,
                          is_stub=cfg["model_factory"].startswith("src.stubs."))
            try:
                inputs = build_messages(task, condition, image_dir=Path(cfg["image_dir"]),
                                        image_source=cfg["image_source"], tools=schemas)
                record.update(messages=inputs.messages, image_ids=inputs.image_ids, tools=inputs.tools)
                if model is None:
                    random.seed(cfg["seed"])
                    torch.manual_seed(cfg["seed"])
                    model = model_factory(model_name)
                record["wrapper_name"] = model.name
                record["wrapper_class"] = type(model).__module__ + "." + type(model).__qualname__
                record["is_stub"] = type(model).__module__.startswith("src.stubs.")
                random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                result = model.generate(inputs.messages, inputs.images, inputs.tools,
                                        cfg["max_new_tokens"], cfg["temperature"])
                # Preserve raw generation even when activation validation fails.
                record.update(text=result.get("text", ""), tool_calls=result.get("tool_calls", []))
                if not isinstance(record["text"], str) or not isinstance(record["tool_calls"], list) or not all(isinstance(c, dict) for c in record["tool_calls"]):
                    raise ValueError("Wrapper must return text:str and tool_calls:list[dict]")
                json.dumps(record)
                hidden = result["hidden_states"]
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or min(hidden.shape) < 1 or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
                    raise ValueError("hidden_states must be finite floating [n_layers, d_model]")
                # Run namespace avoids overwriting activations from other experiments.
                model_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)[:80] + "-" + digest(model_name)[:8]
                activation = output / "activations" / model_slug / activation_namespace / f"{task['task_id']}_{condition}_{repeat}.pt"
                activation.parent.mkdir(parents=True, exist_ok=True)
                temp = activation.with_suffix(".tmp")
                torch.save(hidden.detach().cpu().contiguous(), temp)
                temp.replace(activation)
                record.update(activation_path=str(activation.resolve()), activation_shape=list(hidden.shape),
                              activation_sha256=hashlib.sha256(activation.read_bytes()).hexdigest())
                evidence = classify_outcome(record["text"], record["tool_calls"], tool_condition=inputs.tools is not None, action_tool_names=action_names)
                record.update(rule_outcome=evidence["outcome"], outcome=evidence["outcome"], label_evidence=evidence)
                if judge is not None:
                    record["judge_outcome"] = judge(dict(record))
                    if record["judge_outcome"] not in LABELS:
                        raise ValueError("Judge returned unknown label")
                    record["outcome"] = record["judge_outcome"]
            except Exception as exc:
                # Keep even malformed adapter values inspectable and JSON serializable.
                record = json.loads(json.dumps(record, default=repr))
                record.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
            by_key[key] = record
            persist()
    persist()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for flag in ("tasks", "image-dir", "image-source", "output-dir", "run-dir", "model-factory", "tools-factory", "judge"):
        parser.add_argument("--" + flag)
    for flag in ("models", "conditions", "task-ids", "action-tool-names"):
        parser.add_argument("--" + flag, nargs="+")
    for flag in ("n-repeats", "seed", "max-new-tokens"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--include-drafts", action="store_true", default=None)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path else {}
    config = config or {}
    config.update({k: v for k, v in args.items() if v is not None})
    print(run_eval(config))


if __name__ == "__main__":
    main()
