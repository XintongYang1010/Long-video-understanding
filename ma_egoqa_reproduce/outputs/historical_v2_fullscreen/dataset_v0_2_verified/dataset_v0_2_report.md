# Dataset V0.2 Verified Report

## Counts

- History demo count: 2
- Possible history eval count: 1
- Verified current control count: 12
- Reject count: 5
- Model input rows: 45

## PPT-Ready Case List

- VA_DEMO_001 / DV01_DEMO_001 / Q1078: Why did Shure laugh when Katrina mentioned her voice messages?
- VA_DEMO_007 / DV01_DEMO_007 / Q523: What supported the collaborative mood around the cake task?

## Construction Notes

- Source: Dataset V0.1 plus `visual_audit_v1` human spot-check results.
- No VLM/LLM was run.
- No full-dataset rescreening was run.
- No new videos were downloaded.
- Rejected cases are retained separately for provenance and are excluded from `dataset_v0_2_model_inputs.jsonl`.

## Claim Boundary

Can say:
- Dataset V0.2 contains visually spot-checked candidate cases.

Cannot say:
- Final benchmark complete.
- Labels are final.
- Model accuracy improved.
- Historical memory is proven useful.