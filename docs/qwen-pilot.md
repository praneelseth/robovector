# Qwen pipeline validation and frozen next pilot

The 2026-11-11 Runpod A40 smoke used the pinned Qwen3-VL-8B-Instruct checkpoint
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b, BF16, Torch 2.7.1, Transformers 4.57.1.
Text and image activation comparisons were exactly equal (max difference 0.0).
All twelve saved activation tensors were finite [36,4096] and hash-verified.
The successful Pod was deleted after about 225 seconds; no Pods remained.

## Image audit

See image-audit-20261111.json for the audit of all four image-condition trials.
The authentic 417x313 PNG matches its recorded SHA-256 in every trial. EXIF
orientation did not change dimensions. Processing resized it to 608x448, with
grid [1,28,38] and 266 visual tokens after spatial merging. Prompt metadata
contains image tensors and token positions. No object list, scene description,
or compressed-air identity appeared in the vision messages. This rules out a
missing image or accidental text-only prompt in the recorded preprocessing; it
does not establish visual understanding.

Visual inspection shows the blue/white can at the upper right adjacent to the
gripper. It is small, low-resolution and partly occluded; its pressurized contents
cannot be established confidently from the pixels. Upscaling adds no new detail.
The model's absent-can claims therefore remain perception failures or ambiguity,
not evidence of hazard recognition. No synthetic replacement, crop alteration,
new object labels or claimed power-state information is introduced.

## JSON change

Prompt version integrated-image-only-json-example-v2 adds the same format-only
example with both required keys to every tool condition. It distinguishes place's
relation=on from move_held_object's allowed relations. The parser is unchanged:
missing assistant_text, unknown IDs and invalid enums still invalidate a batch.
Past invalid outputs remain invalid. Local tests establish prompt/schema behavior;
improved Qwen protocol adherence requires the next real-model pilot.

## Frozen next pilot (development data)

configs/pilot_burner_v2.yaml fixes 30 trials: the approved burner pair, three
conditions, five repeats; seed 42, temperature 0.7, 512 output tokens with retries
up to 2048. Use the exact checkpoint above, original real image, BF16/SDPA,
first response, every decoder block, last prompt token, no interventions or
execution. No vision_tools until grounding is agreed. Set only model_dir and
output/run directory for the host. The adjacent freeze manifest hashes inputs.

Do not drop burner based on outcomes. Other task pairs remain drafts; pour remains
excluded. The next pilot tests protocol adherence and repeats image perception
under the unchanged image-only treatment. An image-plus-names diagnostic is a
future separate condition, not a silent alteration to this pilot.

Before interpreting contrasts, require all capture/tensor checks to pass. Report
all schema failures, truncation and perception failures, retaining them in the
specified summary denominators. Hand-review all 30 responses, with automatic and
human labels stored separately. Report counts and denominators; five repeats per
cell are insufficient for strong generalization. If tool formatting remains poor,
pause tool-effect interpretation and revise the protocol in a NEW pilot version.
Do not change the frozen prompts after inspecting its results. No pilot-v2 model
calls have been made yet.

## What the activations mean

A prompt becomes a sequence of tokens (including processed image tokens when
present). At each of 36 language-decoder blocks, each token has a working vector
of 4096 numbers, called its residual-stream representation. Blocks update these
vectors using information from preceding tokens. We record one vector per block
at the LAST prompt position, before the first response token is generated.
That gives a [36,4096] tensor per trial, not all token vectors or model weights.

The last position usually belongs to the chat template's assistant-generation
prefix, not necessarily the last word the user typed. Its exact token ID and
position are saved. Its representation can incorporate the whole allowed prompt,
but it is neither a human-readable explanation nor a dedicated safety score.
Its coordinates are distributed features; one coordinate is not one concept.

Generation first processes the prompt (prefill), then produces response tokens
one at a time. Hooks saved the block outputs during prefill. Independently, a
second forward pass requested hidden states for the SAME prepared prompt. We
selected the same position/layers and compared every number. The final layer was
compared BEFORE final normalization, matching the hook's measurement boundary.
The two tensors were exactly equal for the text and image checks. This validates
our implementation on this checkpoint/environment, not that it measures refusal.
The check temporarily uses two passes; ordinary evaluated trials capture during
generation and need no separate reference pass.

For later analysis, compare many well-controlled refusal/non-refusal examples at
a chosen layer to estimate a candidate refusal direction. A linear harm probe
learns to predict harm labels from vectors. Held-out performance can show encoded
information, but does not by itself show that the information causes refusal.
Causal claims require interventions and controls, currently switched off. Twelve
burner trials alone do not validate either a refusal direction or a harm probe.
