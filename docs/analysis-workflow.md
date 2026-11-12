# Local analysis and optional remote capture

The existing pilot and names diagnostic can be reviewed locally without a GPU,
downloads or another rental. Keep the frozen pilot configuration, integration
source, task file and image unchanged. Rule labels are provisional; manually
distinguish requested harmful steps from numbered safe alternatives.

## Prepare calibration locally

Run from the repository root in PowerShell, using the system Python. The local
validation pools and license notices must already exist under
`data/chat_refusal/downloads/`; this command never downloads them.

```powershell
python scripts/prepare_calibration.py --output outputs/calibration-inputs
python -m pytest tests/test_calibration_inputs.py -q --basetemp tmp/pytest-calibration-inputs
```

Preparation checks validation-pool hashes against `data/chat_refusal/SOURCES.json`
and the presence of local license notices. It reads only `harmful_val.json` and
`harmless_val.json`, samples 32 per label with seed 42, and writes
`calibration.jsonl`, `provenance.json` and a scope README into a fresh directory.
Existing output directories are rejected. Training and test pools are untouched.
The generated directory is ignored by Git. This is exploratory validation data,
not refusal-direction training data or an independent test set. Alpaca-derived
data retain CC BY-NC 4.0 terms for noncommercial research.

## Optional capture on an already authorized Pod

These are reference commands, not authorization to rent or restart a GPU. Copy
the repository and prepared input directory to `/workspace/qwen-smoke`, preserving
relative paths. Use the existing Python environment and pinned local snapshot.
Replace the snapshot directory with the actual existing snapshot path:

```bash
cd /workspace/qwen-smoke
/workspace/smoke-env/bin/python deploy/runpod/calibration_only.py \
  --model-dir /workspace/model-cache/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --calibration-input outputs/calibration-inputs/calibration.jsonl \
  --output /workspace/results/calibration-next --max-seconds 1200
```

For an explicitly authorized new full capture, this runs the frozen 30-trial
pilot, separate 10-trial names diagnostic, then calibration. It requires a fresh
output path. Use either `--model-dir` as above or explicit remote download:

```bash
cd /workspace/qwen-smoke
/workspace/smoke-env/bin/python deploy/runpod/tonight.py \
  --download-on-pod --cache-dir /workspace/model-cache \
  --calibration-input outputs/calibration-inputs/calibration.jsonl \
  --output /workspace/results/full-next --max-seconds 1200
```

The runner validates frozen hashes and calibration provenance, unique safe IDs,
nonempty instructions and balanced labels before loading or downloading a model.
Downloads also require the existing Linux/Runpod/CUDA hardware checks. The model
revision is pinned. `--max-seconds` bounds the loading/inference phase with
generation stop checks; it is not a provider billing cap and excludes download
time. Neither script shuts down the GPU or Pod. Retrieve outputs and handle Pod
termination separately through the authorized lifecycle workflow.

## Preserve paths when retrieving results

Retrieve the entire output directory, including per-trial tensors and raw trial
artifacts, not just the aggregate JSONL. For example, after setting the real SSH
host and port in PowerShell:

```powershell
New-Item -ItemType Directory -Force outputs/runpod/calibration-next | Out-Null
scp -P $PodPort -r "root@${PodHost}:/workspace/results/calibration-next/." outputs/runpod/calibration-next/
```

The aggregate calibration file is `calibration.jsonl`; its tensors are under
`calibration/<id>/prompt_layers.pt`. The pilot uses `pilot/results.jsonl` and the
diagnostic uses `perception_names/results.jsonl`. Saved activation paths refer to
the remote host. The plotting CLI remaps the prefix `/workspace/results/`. For the example
above, use `--artifact-root outputs/runpod` so that the remaining
`calibration-next/...` path matches your local directory. For the existing download, remap
`/workspace/results` to `outputs/runpod/tonight-20261111-l40s/results`.
Do not rewrite the preserved raw result files to fix paths.

## Analyze locally (CPU only)

Install plotting dependencies in your existing project environment:

```powershell
python -m pip install -e ".[analysis]"
```

For the already consolidated November 11 artifacts:

```powershell
python -m src.analysis.refusal_plot --calibration outputs/research/pilot-20261111/calibration.jsonl --results outputs/research/pilot-20261111/plot_trials.jsonl --artifact-root outputs/research/pilot-20261111 --output-dir outputs/research/pilot-20261111/analysis-clean
```

Use a NEW output directory. Outputs: unit direction tensor, numerical scores,
PNG/SVG plot, and hashes for both input JSONL files and every loaded tensor.
Evaluation errors/pending truncation and nonfinite/wrong-shaped tensors are
rejected. Calibration truncation is counted and retained because the contrast is
prespecified by input harm labels and prompt activations precede generation.
Do not infer refusal from an incomplete response. No held-out performance,
validated refusal mechanism or causal claim follows from this plot.

For a new full run saved remotely at `/workspace/results/full-next` and copied
to `outputs/runpod/full-next`, use its `calibration.jsonl` and
`pilot/results.jsonl`, with `--artifact-root outputs/runpod`. Analyze diagnostic
results separately, or make a NEW derived JSONL renaming their condition to
`vision_plus_names` before combining; preserve original run files unchanged.
The analysis package is installed with the project; deployment scripts remain
repository entry points. They require the full checkout, not the small original
smoke ZIP, plus the explicitly prepared calibration inputs. No credentials,
model weights or generated artifacts are committed to Git.
