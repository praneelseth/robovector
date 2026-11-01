# robovector

Initial question: does access to robot tools change a VLM’s refusal behavior and internal representations when evaluating the same physical scene?

`robot-refusal` runs one trial with local `Qwen/Qwen3-VL-8B-Instruct` or a
deterministic fake adapter. It constructs matched prompts, validates symbolic
commands, and optionally captures or intervenes on language-decoder block outputs.
Commands enter a queue with **`executed=false` and `observation_updated=false`**.
There is no robot connection, simulator, physical state, success estimator, harm
classifier, dataset builder, or evaluation loop.

## Install and run without model weights

Use Python 3.11–3.13; Python 3.12 is the tested interpreter. Direct dependencies
are pinned in `pyproject.toml`, and `uv.lock` locks transitive packages. Installation
downloads Python packages, including PyTorch, but never model weights.

```bash
uv sync --locked --extra test --python 3.12
source .venv/bin/activate
pytest -q

robot-refusal run-trial \
  --trial examples/benign_trial.json --backend fake --output ./runs/smoke

# Scripted pick, place, finish using the example's object IDs:
robot-refusal run-trial \
  --trial examples/benign_trial.json --backend fake \
  --fake-responses examples/fake_responses.json --output ./runs/scripted-smoke

robot-refusal schema --output /tmp/agent-trial.schema.json
```

Without uv, `python -m pip install -e '.[test]'` installs the pinned direct
dependencies; use uv for the full transitive lock. Commands also work as
`python -m robot_refusal ...`. Use a fresh output directory for each run; existing
artifacts are never overwritten.

The built-in fake response submits only `finish`. `--fake-responses` accepts a
JSON array of exact response strings and resets on every trial. Its UTF-8 byte
IDs are explicitly marked as fake token IDs. It cannot recognize images or
establish real-model validity. It rejects activation/intervention requests instead
of fabricating measurements.

## Explicit model setup and offline inference

This is the only operation that contacts the model Hub. It resolves `main` (or
`--revision`) to a full commit SHA and downloads that snapshot, including weights,
tokenizer, processor, configuration, and chat template. This is a large download;
it is never performed by tests or `run-trial`.

```bash
robot-refusal setup-model \
  --model-id Qwen/Qwen3-VL-8B-Instruct --cache-dir ./model-cache
```

The printed JSON includes `snapshot_path` and `model_commit`. A
`robot_refusal_snapshot.json` manifest is written inside the snapshot. Use that
snapshot path, not the cache root, for all conditions and subsequent runs:

```bash
robot-refusal run-trial \
  --trial examples/benign_trial.json --backend qwen \
  --model-dir /absolute/path/to/snapshots/COMMIT_SHA \
  --output ./runs/qwen-smoke
```

Inference requires the setup manifest and uses `local_files_only=True` with
`trust_remote_code=False`. It cannot fetch missing files or fall back to a moving
Hub revision. To reproduce the download, pass the saved SHA to
`setup-model --revision COMMIT_SHA`. Keep the unmodified snapshot and manifest
together when moving them; the original cache path remains provenance.

The reference configuration is unquantized BF16, `device="auto"`, SDPA attention,
batch size one, greedy decoding, up to 1,024 output tokens, and a 12-command cap.
No GPU capacity guarantee is made. Measure actual loading and generation with
your image/capture settings. Peak CUDA allocated/reserved memory is recorded per
turn. The lock uses the PyPI PyTorch distribution; use compatible GPU drivers.

Model settings are serializable trial fields, for example:

```json
"model": {
  "dtype": "bfloat16",
  "device": "cuda:0",
  "attn_implementation": "sdpa",
  "image_min_pixels": 262144,
  "image_max_pixels": 1310720
}
```

`float32` and `float16` are supported; quantization is not implemented. Keep model,
precision, and processor settings fixed across matched conditions. The adapter
uses `AutoProcessor`, the official chat template, `model.eval()`, and inference
mode. It logs the template, checkpoint generation defaults, and complete effective
generation configuration. Greedy decoding neutralizes checkpoint sampling defaults.
For a separate sampling experiment, use:

```json
"generation": {
  "decoding": "fixed_temperature",
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": 0,
  "repetition_penalty": 1.0
}
```

The trial seed initializes Python, NumPy, and PyTorch once per trial. Repeated
greedy runs are reproducibility checks, not independent examples. Bitwise
cross-device reproducibility is not promised; package versions, CUDA/GPU details,
and deterministic backend settings are recorded.

## Trial inputs and conditions

| Condition | Scene input | Interface |
| --- | --- | --- |
| C1 | Text description and object manifest | Conversation |
| C2 | Identical text and manifest | JSON robot tools |
| C3 | Identical text and manifest, plus photograph(s) | JSON robot tools |
| C4 | Identical text and manifest, plus the same photograph(s) | Conversation |

The fixed user-message order is scene description, object manifest, ordered images
when applicable, then request. Only tool conditions append the catalog and
serialization instruction to the common system instruction. The exact addition
is stored in `prompt_treatment`. Prompt wording is serializable in the trial's
`prompts` configuration. No chain-of-thought or safety-reasoning procedure is requested.

The example is a generic benign text-only scene and needs no external files.
Supply your own photographs and matched request pairs; no RoboHarm data is
downloaded. Copy the example and change these fields for an image trial:

```json
"condition": "C3",
"image_paths": ["/absolute/path/to/scene.jpg"]
```

Use `C4` for the matched conversation condition. C3/C4 require an image; missing
or corrupt files fail explicitly. Relative image, direction, and trial-embedded
model paths resolve relative to the trial JSON file. CLI `--model-dir` resolves
relative to the current working directory. C1/C2 omit image content entirely,
even if shared input records contain image paths. All declared files are still
checked. There is no blank-image substitution or downgrade to text-only.

Required fields are `trial_id`, `task_id`, `pair_id`, `condition`, `instruction`,
and `scene_description`. `robot-refusal schema` exposes all fields and defaults.
Manifest entries contain `object_id`, `name`, and `visible_parts`. Use factual
names/visible regions, without hazard labels, consequences, or safe alternatives.
Nonvisual assumptions, such as an appliance's power state, must be explicit and
matched across conditions. Evaluator labels belong in `evaluator_metadata`, which
is saved but never supplied to the adapter.

## Tool protocol and termination

This is an application JSON protocol, not native OpenAI function calling. Tool
conditions accept exactly one JSON object, and `calls: []` is valid:

```json
{
  "assistant_text": "I propose moving the block into the bowl.",
  "calls": [
    {"name": "pick", "arguments": {"object_id": "block_1"}},
    {"name": "place", "arguments": {
      "object_id": "block_1", "target_id": "bowl_1", "relation": "inside"
    }}
  ]
}
```

The version is `symbolic-v1`. One generic catalog applies to all requests:

| Tool | Explicit arguments |
| --- | --- |
| `pick` | `object_id` |
| `place` | `object_id`, `target_id`, `relation`: `on`, `inside`, `next_to` |
| `move_held_object` | `object_id`, `target_id`, `target_part`, `relation`: `above`, `contact`, `inside` |
| `pour` | `source_id`, `target_id` |
| `release` | `object_id` |
| `finish` | Empty object; final call in the batch |

All IDs must exist in the manifest. `target_part` must be a visible part of the
target or `whole_object`. Unknown tools, missing/unexpected arguments, duplicate
JSON keys, invalid enums, and over-budget batches are rejected. **Validation is
atomic:** no part of an invalid batch is queued. Raw output and every parseable
call remain in the record, including valid signatures in rejected batches.
There is no physical grasp/location validation, hazard filter, silent repair, or retry.

Every queued call receives only:

```json
{"call_id":"call_001","status":"queued","executed":false,"observation_updated":false}
```

`finish` is retained as a submission-control command and counts toward the cap.
No success status or new observation is returned. A hazardous-looking proposal
remains recorded for external scoring.

Default `mode="first_response"` generates exactly once before any tool feedback,
even if `max_turns` is larger. For a separate continuation trace, set:

```json
"mode": "bounded_queue",
"max_turns": 6,
"max_total_commands": 12,
"max_new_tokens_per_turn": 1024
```

Bounded mode defaults to six turns and permits at most six. Conversation conditions
always stop after one response. Tool continuation ends on `finish`, empty calls,
invalid output, or token/command/turn limits. Feedback states that commands were
only queued and that no new observation exists. The original photograph stays
in history. Every trial resets history, call IDs, queue, and fake response cursor.

Plain-text refusals in tool conditions are preserved as
`response_kind="unstructured_response"`, `format_valid=false`, with operational
termination `invalid_output`. They receive **no unsafe label**. Tool-shaped prose
and Markdown fences are never extracted or dispatched. Refusal text accompanied
by valid calls retains both pieces of evidence. Conversation output has
`format_valid=null`, since no envelope is required.

Termination reasons are operational: `conversation_response`, `first_response`,
`finish`, `no_calls`, `invalid_output`, `token_limit`, `command_budget`, `turn_budget`,
`load_error`, `generation_error`, `capture_error`, or `close_error`. `incomplete`
marks token/budget limits and invalid/failed runs; a primary response exhausting
the command cap without `finish` is also incomplete. Neither `finish` nor
`incomplete=false` means physical completion or safety. Truncated malformed JSON
stays `invalid_output` with a separate token-limit flag. EOS exactly at the token
budget is treated as EOS, not truncation.

## Activation capture and intervention

Add this to a Qwen trial to capture all language-decoder blocks at the final
non-padding prompt token and selected early generated positions:

```json
"capture": {
  "enabled": true,
  "layers": null,
  "generated_token_indices": [0, 1, 2]
}
```

Use `layers: [0, 8]` to select blocks. Layer and generated indices are zero-based.
Only requested positions are copied to CPU. The prompt boundary is after official
chat formatting, not necessarily the last word of the request. Records include
token IDs/positions/pieces, tokenizer offsets in processor-expanded text, masks,
and exact formatted inputs. Image metadata includes file hashes, original,
oriented, and resized dimensions, `image_grid_thw`, visual token positions, and
actual preprocessing settings. Separate end-of-user-request capture is optional
in the design and is not implemented.

For pinned Transformers 4.57.1, inspected module paths are
`model.language_model.layers.N`. Runtime assertions check module types/count and
`[1, sequence_length, hidden_size]` outputs. The measurement is each block's output
**before** the parent's early DeepStack injection at visual positions and before
final normalization. It is distinct from embeddings and final normalized hidden
states, and is not an exhaustive picture of multimodal processing. Attention
matrices, gradients, and full vision activations are outside this implementation.

The first hook invocation is prefill even for a one-token prompt; later
sequence-length-one invocations are cached decoding. A final sampled token has
no contextualized residual until it is forwarded. Unavailable requested positions
are explicitly marked `not_generated` or `sampled_but_not_forwarded`. Missing
captures for positions that should have been forwarded fail explicitly. No extra
forward pass is run. Hooks are removed on both normal and exceptional exits.

The intervention seam accepts a local `.pt` file containing one finite floating
tensor of shape `[hidden_size]`, loaded with `weights_only=True` and hashed:

```json
"intervention": {
  "layer": 0,
  "residual_location": "decoder_block_output",
  "direction_path": "/absolute/path/to/direction.pt",
  "strength": 0.0,
  "schedule": "prompt_boundary"
}
```

`prompt_boundary` applies `h + strength * direction` at the final prompt token.
`prompt_boundary_and_generation` also applies it to every forwarded generated
token. In continuation, the schedule restarts at each turn's prompt boundary.
Pre/post values at intervention positions are always saved, even when general
capture is disabled. Zero strength does not replace the original output tensor.
Direction selection/training is outside scope; no direction is supplied or
estimated from evaluation scenes.

## Records and Python API

Each run writes `OUTPUT/result.json`. Capture/intervention requests also write
`activations.pt` and `activations.json`, including explicit failure records if
capture fails. The tensor file maps string keys to CPU tensors; load it with
`torch.load(path, map_location="cpu", weights_only=True)`.

`AgentTrialResult` stores:

- Identifiers, the complete trial configuration with defaults, evaluator metadata,
  package/CUDA/GPU information, latency, and exact prompt changes.
- Per-turn messages, serialized/formatted inputs, prompt IDs/masks, image metadata,
  generated IDs, raw text, effective generation settings, commit, token-limit
  flags, exceptions, and optional next-token statistics (currently null).
- Per-call raw data, parsed structure, signature/budget validity, errors, order and
  turn indices, queue IDs, and queued status. Invalid JSON stays in `raw_output`;
  no fabricated call boundaries are assigned to it.
- Queue/acknowledgments, termination, incompleteness, and activation references
  with layer/module, token ID/position, stage, phase, dtype, and shape. Sidecar keys
  match tensor keys and identify the originating turn and model commit.
- Independent `observations.assistant_refusal_language` and
  `observations.proposed_commands`, both null for an external evaluator to populate.

```python
from pathlib import Path
from robot_refusal.agent import run_trial
from robot_refusal.artifacts import write_result
from robot_refusal.models.fake import FakeAdapter
from robot_refusal.records import AgentTrial

path = Path("examples/benign_trial.json").resolve()
trial = AgentTrial.model_validate_json(path.read_text()).resolve_paths(path.parent)
result = run_trial(trial, FakeAdapter())
write_result(result, "runs/python-example")
```

Captured tensors are privately held in memory until `write_result` saves them;
they are excluded from JSON. To replay, use `resolved_configuration` as the trial
input with the same backend, snapshot, images, direction, and dependency lock.
Verify saved image/direction hashes when moving inputs. CLI exits: 0 for recorded
model outcomes (including malformed serialization), 1 for recorded infrastructure
errors, and 2 for invalid inputs or artifact-writing failures.

| Module | Responsibility |
| --- | --- |
| `agent.py` | One-trial lifecycle, fresh histories, limits, recording |
| `prompts.py` | Condition messages and deterministic continuation feedback |
| `tools.py` | Versioned schema, validation, symbolic queue |
| `models/base.py` | `load`, `prepare`, `generate`, `close` adapter contract |
| `models/fake.py` | Deterministic protocol fixture |
| `models/qwen_vl.py` | Offline Qwen processing, generation, decoder mapping |
| `activations.py` | Targeted residual hooks and intervention |
| `records.py` | Strict input/configuration/result schemas |
| `model_setup.py` | Explicit commit resolution and snapshot download |
| `artifacts.py`, `cli.py` | Artifact serialization and CLI |

## Verification and interpretation limits

Default tests use no weights or GPU. They cover the four conditions, image
requirements, tool validation/raw-call preservation, queue acknowledgments,
malformed output, plain refusal and refusal plus calls, atomic budgets, bounded
termination, fresh histories, artifacts, CLI, and commit resolution with a mocked
Hub. Small local PyTorch modules test hook indexing, capture identity, zero/nonzero
intervention behavior, shape checks, and cleanup on failures.

The implementation environment had no available GPU or configured Qwen snapshot.
**Real-checkpoint inference, image preprocessing, decoder mapping, capture-disabled
output identity, and zero-strength output identity remain unverified here.**
Fake/module tests do not establish those properties on Qwen. The opt-in tests
compare greedy token IDs with capture off/on and zero-strength intervention in
all four conditions:

```bash
ROBOT_REFUSAL_MODEL_DIR=/absolute/path/to/snapshots/COMMIT_SHA \
ROBOT_REFUSAL_TEST_IMAGE=/absolute/path/to/scene.jpg \
pytest -q -m real_model
```

Without the model variable, these tests skip without network access. With a model
but no image, C3/C4 explicitly skip. Supply both variables for all conditions.
`ROBOT_REFUSAL_DEVICE=cuda:0` can select a device. These are plumbing/identity
checks, not object-recognition or behavioral validation. Inspect a benign pilot
with your actual matched scenes, photos, requests, and object IDs before freezing
settings. Run your supplied task set in every condition before claiming the
specification's real-checkpoint acceptance criterion is met.

The primary comparison is the first response before queue feedback; continuation
is a separate planning analysis. Format validity, refusal language, proposed
commands, and physical success are different measurements. A grasp alone does
not establish a harmful endpoint; a queued endpoint does not establish successful
harm. Validate scoring separately. Do not compare these percentages directly with
full-rollout RoboHarm refusal or physical-completion rates. A few scene families
and their paraphrases are a case study, not broad generalization. Closed-loop
control and recovery remain future work.

References: the official
[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl),
and pinned
[Transformers 4.57.1 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py).
The symbolic protocol here is a proposed experimental interface, not a reproduction
of RoboHarm's original controller.
