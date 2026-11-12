# Development pilot and exploratory contrast, 2026-11-11

The frozen 30-trial burner-v2 pilot is unchanged. Its hashes are verified on the
GPU before model loading. Five repeats per label and condition, seed/config and
capture boundaries remain as frozen. No interventions or robot execution.

The additional perception run is a SEPARATE artifact with a distinct prompt
version. It repeats only vision, five repeats per label, adding exactly
`objects: compressed air can, burner.` as another user text part. The same original
image and instructions are used. Seeds match the frozen vision trials, but the
extra text changes tokenization, prompt length, and last-token context. Compare
responses and label reasons, not merely generic refusal counts. Names provide
semantic hazard information as well as helping identify objects; a changed
response is not a clean causal proof that perception alone explains the gap.

With explicit user approval, the calibration uses 32 harmful and 32 harmless
VALIDATION prompts sampled with seed 42 from the licensed local Arditi pools.
The mixed harmful training pool remains withheld, and test splits are untouched.
These validation prompts are now exploratory calibration data, not held-out
evaluation. This is not replication of Arditi's training procedure. Plain user
chat, no image/tools, greedy output up to 128 tokens; raw outputs/truncation are
saved to assess whether harm labels actually elicit refusals. Truncated calibration
outputs have no behavioral label, but their pre-generation activations remain
well-defined and are included in the harm-label mean contrast as prespecified.

At each layer compute mean(harmful activations) minus mean(harmless activations),
normalize to unit length, and dot each burner's last-prompt vector with that layer's
direction. Plot all layers, one line per condition; harmful and harmless burner
trials have separate panels. Do not choose a layer or flip signs to favor a result.
This is an exploratory harm contrast, not a validated causal refusal direction.
Prompt wording, length, formatting, residual norms and semantic content confound
cross-condition projections. Repeated trials have the same prompt; prefill
activations may be identical across seeds, so repeats are not independent prompt
samples. Lines separating is a lead for controlled experiments, not a conclusion.

The doll pair stays draft. Its safe twin preserves stabbing and scene/implement,
but changes target geometry and location. The original harmful phrase "the thing
that's not the bread" is ambiguous (the knife is also not bread). Doll harm is
symbolic violence, not observed injury to a person. Retain paper wording and seek
explicit approval before adding this pair to any frozen evaluation.
