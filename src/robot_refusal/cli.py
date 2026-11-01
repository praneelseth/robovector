"""Minimal single-trial CLI; explicit setup is separate from offline inference."""

import argparse
import json
from pathlib import Path
import sys

from .records import AgentTrial, MODEL_ID


def parser():
    root = argparse.ArgumentParser(prog="robot-refusal")
    commands = root.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup-model", help="Resolve and download a pinned Qwen snapshot (large download)")
    setup.add_argument("--model-id", default=MODEL_ID, choices=[MODEL_ID])
    setup.add_argument("--cache-dir", required=True)
    setup.add_argument("--revision", default="main")
    run = commands.add_parser("run-trial", help="Run one trial and write JSON plus optional activation tensors")
    run.add_argument("--trial", required=True)
    run.add_argument("--backend", choices=["fake", "qwen"], default="fake")
    run.add_argument("--model-dir")
    run.add_argument("--output", required=True)
    run.add_argument("--fake-responses", help="JSON array of exact response strings for deterministic protocol tests")
    schema = commands.add_parser("schema", help="Print the complete AgentTrial JSON Schema")
    schema.add_argument("--output")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "setup-model":
            from .model_setup import setup_model
            print(json.dumps(setup_model(args.model_id, args.cache_dir, args.revision), indent=2))
            return 0
        if args.command == "schema":
            schema = json.dumps(AgentTrial.model_json_schema(), indent=2) + "\n"
            if args.output:
                Path(args.output).write_text(schema)
            else:
                print(schema, end="")
            return 0
        trial_path = Path(args.trial).expanduser().resolve()
        trial = AgentTrial.model_validate_json(trial_path.read_text()).resolve_paths(trial_path.parent)
        if args.model_dir:
            data = trial.model_dump()
            data["model"]["model_dir"] = str(Path(args.model_dir).expanduser().resolve())
            trial = AgentTrial.model_validate(data)
        trial.check_input_files()
        output = Path(args.output)
        if any((output / filename).exists() for filename in ("result.json", "activations.pt", "activations.json")):
            raise FileExistsError(f"Refusing to overwrite trial artifacts in {output}")
        if args.backend == "fake":
            from .models.fake import FakeAdapter
            responses = json.loads(Path(args.fake_responses).read_text()) if args.fake_responses else None
            if responses is not None and (not isinstance(responses, list) or not all(isinstance(r, str) for r in responses)):
                raise ValueError("--fake-responses must contain a JSON array of strings")
            adapter = FakeAdapter(responses)
        else:
            if args.fake_responses:
                raise ValueError("--fake-responses can only be used with --backend fake")
            from .models.qwen_vl import QwenVLAdapter
            adapter = QwenVLAdapter()
        from .agent import run_trial
        from .artifacts import write_result
        result = run_trial(trial, adapter)
        destination = write_result(result, args.output)
        print(json.dumps({"result": str(destination.resolve()), "termination_reason": result.termination_reason,
                          "queued_commands": len(result.command_queue), "incomplete": result.incomplete}))
        # Invalid serialization is a measured model outcome, not an infrastructure failure.
        return 1 if result.exceptions else 0
    except Exception as exc:
        print(f"robot-refusal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
