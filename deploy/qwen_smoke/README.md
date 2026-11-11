# qwen-smoke

Temporary Baseten test of the pinned Qwen3-VL-8B checkpoint and integrated runner.
The user authorized a $5 maximum and remote-only weight download. The model name
is deliberately generic. No shared Model API inference or other projects are used.

The first attempt on 2026-11-03 UTC was rejected by Baseten with:
`You must add a payment method to deploy models.` A subsequent model-list query
returned an empty list; no inference request was sent and no GPU result exists.
The local record is outputs/baseten/qwen-smoke-20261103/blocked-report.json.
Workspace credits remain unverified; billing-summary requests timed out.

## Prepared experiment

- Qwen/Qwen3-VL-8B-Instruct, revision in checkpoint.json; 17.53 GB weights.
- One A100, BF16, pinned Transformers 4.57.1 / PyTorch 2.7.1.
- Weights downloaded by Baseten's weights mount, never onto the laptop.
- Independent capture equality checks for text and the authentic burner image.
- Approved burner pair, text_chat/text_tools/vision, two repeats: 12 trials.
- Shared symbolic JSON tools, no execution. vision_tools omitted pending grounding.
- 256 initial output tokens; bounded truncation retries up to 1,024.
- One loaded model reused across trials. Eight-minute inference deadline.
- All raw trial files, labels, tensors and capture-check reports returned as a ZIP.
- Repeating the fixed HTTP operation returns the cached result, without new generation.

## Spending controls

The published A100 rate checked for this attempt is $0.06667/minute (~$4/hour).
The controller starts an independent local watchdog BEFORE creating the model.
It requests deactivation at 30 minutes from launch, including build time, or
immediately after inference/failure. It configures min replicas=0, max replicas=1,
and a 60-second scale-down delay. Expected compute at the cutoff is about $2,
leaving margin within $5 for shutdown latency. These are operational controls,
not a provider-enforced dollar cap: losing the host/network can delay explicit
deactivation, so verify the provider's stopped state. Never leave the test unattended.
Do not create another replica/deployment or switch hardware to work around failure.

## Resume after workspace billing is enabled

Use a fresh session folder and the isolated client environment (requirements-client.txt).
The env-file argument points to the existing authorized credential location; it is
read in memory, not copied into the upload or written to a new auth configuration.

```powershell
python scripts/baseten_smoke.py prepare --folder outputs/baseten/<new-session>
python scripts/baseten_smoke.py deploy --folder outputs/baseten/<new-session> --env-file <existing-env-file>
python scripts/baseten_smoke.py status --folder outputs/baseten/<new-session> --env-file <existing-env-file>
# Invoke once after status is ACTIVE and controls are verified:
python scripts/baseten_smoke.py invoke --folder outputs/baseten/<new-session> --env-file <existing-env-file>
python scripts/baseten_smoke.py down --folder outputs/baseten/<new-session> --env-file <existing-env-file>
```

The inference command always requests shutdown in its finally block. Verify zero
active replicas and INACTIVE afterward. Do not automatically retry an ambiguous
network failure. The guard handles a push that creates a model but fails before
returning its ID by discovering only this exact name and creation-time window.

The upload is an explicit allowlist assembled under the ignored output directory,
not a recursive upload of the repository. It contains the necessary Python modules,
two task records, one authentic image and pinned checkpoint metadata. It excludes
credentials, other projects, previous outputs, local environments and weights.
