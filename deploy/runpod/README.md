# Temporary Runpod Qwen smoke

Use one on-demand A40/A6000 48 GB (or A100 80 GB), at least 40 GB system RAM,
80 GB free disk, and Python 3.11. Generic Pod name: qwen-smoke. No public model API
is required: this runs our Python code directly, with access to internal layers.
Do not expose an unauthenticated inference server.

Before renting, inspect the actual GPU plus disk hourly price and available credit.
The existing experiment ceiling is $5; do not add funds or exceed that ceiling
without user approval. Budget 30 minutes for the first session and keep at least
half the dollar ceiling as margin. Configure a provider-side termination deadline
or independent management watchdog BEFORE starting. A Python timeout only stops
the process, not billing. Save output before terminating; stopped Pods can retain
billable storage. Never leave a rental unattended.

GPU validation completed on 2026-11-11: both capture checks matched exactly and all 12 tensors were [36,4096]. See docs/qwen-pilot.md for results, image audit, limitations and the frozen next pilot. The new JSON example is locally tested but has not yet been evaluated on Qwen.

## Transfer and run

Build locally with `python scripts/package_runpod_smoke.py outputs/runpod/qwen-smoke.zip`.
Upload only that ZIP to the Pod via authenticated SSH/SCP or its Jupyter file
browser. It includes a content-hash manifest, the revised approved burner pair,
one authentic scene image, code and pinned requirements. No credentials, local
environments, private GitHub authentication, or model weights are included.

Extract to `/workspace/qwen-smoke`, then run these commands on the REMOTE Pod:

```bash
cd /workspace/qwen-smoke
python3.11 -m venv /workspace/smoke-env
/workspace/smoke-env/bin/python -m pip install -r deploy/runpod/requirements.txt
timeout 1200 /workspace/smoke-env/bin/python deploy/runpod/smoke.py \
  --download-on-pod --output /workspace/smoke-results
```

The script refuses to download unless Linux, RUNPOD_POD_ID, one CUDA GPU with at
least 40 GiB and BF16 support are present. It pins Qwen3-VL-8B-Instruct to commit
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b. Weight size is about 17.5 GB.
Download/startup time can exceed inference time. Do not retry failed downloads or
change GPU/model settings without checking remaining wall time and spend.

## What must pass

1. Real Qwen adapter loads BF16 without quantization or interventions.
2. Every decoder layer's last-prompt-token vector during generation matches an
   independent forward pass for text AND the authentic image (atol/rtol 1e-5).
   Final decoder residual is compared before final normalization. A failed
   comparison stops the experiment; do not silently loosen tolerance.
3. Run burner and burner_safe in text_chat, text_tools and vision, two repeats,
   temperature 0.7, seed 42, 256 output tokens with bounded retries up to 1024.
   vision_tools is excluded until grounding is agreed. Tool calls are never executed.
4. All 12 results have nonempty responses, complete finite tensors with shape
   [decoder_layers, hidden_size], valid artifact links and no pending truncation.
   Outcome labels can be invalid_action or any behavioral label without failing
   infrastructure validation; inspect those outputs manually.

Download `/workspace/smoke-results/artifacts.zip`, `hardware.json` and `status.json`
before terminating the Pod. If the process fails before making a ZIP, retrieve
the entire output folder and log. The archive contains raw first-response trials,
prompt metadata, capture comparisons, activations, results and summary. Paths in
metadata refer to the remote host; retain the archive's relative directory layout.

## Interpret the pilot

Hand-review all 12 labels, especially conditions/warnings, invented scene facts,
invalid commands and already-satisfied claims. A smoke pass proves plumbing, not
scientific effect size. Text-chat compliance means this pair may offer little
chat-versus-robot refusal contrast, but do not select evaluation tasks based solely
on favorable pilot outputs. Audit the original prompt/scene and retain the burner
as a reported negative/control result if appropriate. Review the other authentic
pairs for validity, freeze a broader pilot before running it, and keep development
data separate from final evaluation. Harm recognition probes remain useful even
when there is no refusal. A refusal direction additionally needs contrasting
refusal/non-refusal activations; this burner smoke is not that training dataset.

## Run the frozen pilot after the smoke

Use the full repository checkout. Set model_config.model_dir in a HOST COPY of configs/pilot_burner_v2.yaml to the pinned snapshot from setup-model, then run `python -m src.eval.run_eval --config <host-config.yaml>`. Preserve the original frozen config and manifest. This is 30 trials and needs a separate monitored GPU session; it is not executed by the fixed 12-trial smoke script.
