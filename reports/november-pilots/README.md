# RoboVector development experiments

These analyses use Qwen3-VL-8B-Instruct. The staged results below summarize the local experiment log. Plots and selected numeric summaries are included here. Raw run archives, activation tensors, and review packets remain local.

| Stage | Result |
|---|---|
| [12. Fixed response choice](refusal-choice-control/choice_margins.png) | Candidate ablation reduced the number of harmful prompts favoring refusal from 24/24 to 13/24. Matched random ablation kept it at 24/24. This is a forced-choice result against one random draw. |
| [13. Position audit](harm-refusal-position-audit/paired_projection_gaps.png) | The instruction-end calibration direction reversed the harm ordering across scene arrangements at block 18. It was unsuitable as a gate on these controls. |
| [14. Perfect-gate replay](selective-tool-safety-development/offline_oracle_counts.png) | Applying the existing addition only on hazardous requests still left 2/5 accepted hazardous plans and 2/5 accepted safe requested plans. This replay reused saved responses. |
| [15. Initial tool choices](tool-abstention-choices/abstention_margins.png) | No tested edit flipped a hazardous case from proceed to abstain. The reference edit applied a smaller displacement than fixed addition. |
| [16. Wording and order](tool-choice-robustness/wording_margins.png) | Matched scene ranking held after averaging orders, but the original burner hazard choice flipped with A/B order. A/B was never the top token under the competing JSON instructions. |
| [17. A/B format check](tool-choice-format/answer_order_counts.png) | After placing the A/B requirement in the system prompt, A/B was the top token in all 120 passes. Hazardous matched scenes scored higher in 12/12 checks in each order. The original burner favored dangerous and proceed in all six checks. Absolute classification at zero remained order-sensitive. |
| [18. Hazard veto replay](tool-hazard-cutoff/plan_replay.png) | Leaving one development family out at a time, a separate hazard veto would have blocked 3/3 previously accepted hazardous plans and 1/4 accepted safe plans. The [frozen development cutoff](tool-hazard-cutoff/cutoff_freeze.json) awaits independent families. |
| [19. JSON abstention protocol](symbolic-json-abstention/protocol_outcomes.png) | Across five inert symbolic pairs, explicit abstention reduced forbidden placements from 2/5 to 0/5 and allowed placements from 5/5 to 3/5. All 20 outputs passed validation. Correct rule-based abstention improved from 3/5 to 4/5. |

The Stage 11 free-generation random-ablation attempt stopped before producing responses. Stage 12 uses forward scores instead. Stages 12 and 15–17 do not generate plans. Stage 19 generates inert symbolic plans with no actuators or real hazards and does not establish performance on the original benchmark.

Next, apply the frozen cutoff once to the new independently reviewed families and compare safe-plan retention and hazardous-plan blocking with explicit abstention. Review the original blinded tool labels before treating them as final. Learned gating still needs a steering intervention that improves plan outcomes when applied to the correct cases.
