# RoboHarm image provenance

Checked 2026-11-01. No standalone scene/image release was located in the sources checked; this is not a claim that no release exists anywhere.

- Inspect Robots framework: https://github.com/robocurve/inspect-robots at `c8a821ba7bd44c042c95ab8f6a83d1e776cfb4a3`. Recursive file listing and README inspected; no RoboHarm scene/image assets located. Framework license: MIT (does not license unrelated images).
- WorldEvals: https://github.com/robocurve/worldevals at `c6814627c9c2f57984b9eb08757142f7e26b653f`, catalog source and https://worldevals.org/ inspected. The live catalog lists VibeCheckBench, KitchenBench and StationeryBench, not RoboHarm.
- Paper abstract, HTML and PDF inspected: https://arxiv.org/abs/2610.14043 and https://arxiv.org/html/2610.14043v1. Paper links to Inspect Robots, with no separate project/asset release located.

## Source and transformation

All PNGs below are crops of Figure 1, PDF page 2, in Sun et al., *Refuses in Chat, Complies with Arms: LLM Refusal Does Not Survive Robot Control* (2026).
Source PDF: https://arxiv.org/pdf/2610.14043v1
Authors: Edward Sun, Christopher Leet, Sravanthi Machcha, Achyutha Menon, Tzu Kit Chan, Yuchen Cui, Jay Chooi.
License displayed on arXiv: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Modification: crop only. No generated images, semantic edits, sharpening or upscaling.

The PDF embeds Figure 1 as a 2200 x 736 raster. Crop coordinates below are in that original raster, origin top-left, `(left, top, right-exclusive, bottom-exclusive)`. Its PDF placement is approximately `(159.479, 72.000, 452.527, 170.038)` points on page 2 (612 x 792 points). Real is the upper row; sim is the lower row.
Each output is 417 x 313 pixels. This is publication-resolution evidence, not an original camera capture; labels and power states may be unreadable. The same image IDs are shared by instruction-varied twins.

Reproduce: `python scripts/extract_roboharm_images.py path/to/2610.14043v1.pdf`
Requires the `dev` extra (PyMuPDF). `crops.json` stores machine-readable coordinates and SHA-256 hashes. All ten requested views were recovered and visually checked.

| File | Figure / page | Crop box |
|---|---|---|
| doll_real.png | 1 / 2 | `[77, 66, 494, 379]` |
| doll_sim.png | 1 / 2 | `[77, 411, 494, 724]` |
| burner_real.png | 1 / 2 | `[502, 66, 919, 379]` |
| burner_sim.png | 1 / 2 | `[502, 411, 919, 724]` |
| toaster_real.png | 1 / 2 | `[928, 66, 1345, 379]` |
| toaster_sim.png | 1 / 2 | `[928, 411, 1345, 724]` |
| brick_real.png | 1 / 2 | `[1353, 66, 1770, 379]` |
| brick_sim.png | 1 / 2 | `[1353, 411, 1770, 724]` |
| pour_real.png | 1 / 2 | `[1779, 66, 2196, 379]` |
| pour_sim.png | 1 / 2 | `[1779, 411, 2196, 724]` |

Source PDF SHA-256: `41b0897c70675a9c80b3ae3cd51f5297e11f228f0d12db9ea394639894cbe903`.
