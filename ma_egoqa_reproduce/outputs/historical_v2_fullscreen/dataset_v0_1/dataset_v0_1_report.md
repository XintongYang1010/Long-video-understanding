# Dataset V0.1 Report

Dataset V0.1 contains auto-screened candidate cases for evidence-scope comparison. It repurposes MA-EgoQA/EgoLife caption evidence and is not a final benchmark.

## Input Full Screening Counts

- Processed questions: 1739.
- Tier A: 22.
- Tier B: 111.
- Tier C: 1190.
- current+historical better than current-only: 73 yes + 95 unclear_but_promising.

## Dataset Counts

- demo set count: 10.
- eval-lite set count: 35.
- control set count: 15.
- model input JSONL rows: 150.

## Sources

- demo set sources: {'tier_A_demo_ready': 10}.
- eval-lite set sources: {'tier_A_demo_ready': 15, 'tier_B_promising_history_gain': 20}.
- control set sources: {'tier_C_current_sufficient_control': 15}.

## Recommended Human Spot-Check

- DV01_DEMO_001 / Q1078 / current_plus_history_gain / high / Why did Shure laugh when Katrina mentioned her voice messages?
- DV01_DEMO_002 / Q617 / current_plus_history_gain / high / What decision-making role did Jake take in the group task involving glasses?
- DV01_DEMO_003 / Q313 / current_plus_history_gain / high / When Katrina discussed her breakup, how did the group mostly respond?
- DV01_DEMO_004 / Q1040 / current_plus_history_gain / high / Who incorrectly assumed everyone would want an auction group of four items?
- DV01_DEMO_005 / Q982 / current_plus_history_gain / high / Why didn't Katrina seem fully involved in the conversation about renting costumes during 18:23?
- DV01_DEMO_006 / Q1073 / current_plus_history_gain / high / Why did Jake smile and respond to Nicous by referring to a 'national guardian'?
- DV01_DEMO_007 / Q523 / current_plus_history_gain / high / What supported the collaborative mood around the cake task?
- DV01_DEMO_008 / Q437 / current_plus_history_gain / high / How did Jake contribute to the group's tasks during the time Shure prepared coffee?
- DV01_DEMO_009 / Q433 / current_plus_history_gain / high / Why did they decide on having transition shots in the choreography?
- DV01_DEMO_010 / Q419 / current_plus_history_gain / high / How was coordination handled in dividing cleanup responsibilities?

## Why This Only Shows Feasibility

- The cases are selected by caption-level retrieval and heuristic answerability, not by final human annotation.
- The original MA-EgoQA questions were not designed specifically for this self-first historical-memory idea.
- Evidence gain here means candidate evidence coverage gain, not model correctness or causal proof.
- Each case keeps `label_status=auto_screened_needs_human_check`.

## Current Claims

- Dataset V0.1 contains auto-screened candidate cases for evidence-scope comparison.
- The dataset includes candidate cases where current+historical evidence may improve coverage over current-only evidence.

## Claims Not Supported

- Final labels.
- Benchmark complete.
- Method works.
- Accuracy improved.
- Historical memory proven useful.

## Next Model Comparison Step

1. Use `dataset_v0_1_model_inputs.jsonl` to create three evidence-scope prompts per case: current_only, history_only, and current_plus_historical.
2. Compare model outputs for current_only vs current_plus_historical under the same prompt and decoding settings.
3. Treat demo/eval-lite results as evidence-scope diagnostics only until manual spot-check confirms the cases.
4. Use control cases to verify the system can stop at current evidence when current-only is already sufficient.
