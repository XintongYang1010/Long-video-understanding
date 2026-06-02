# Dataset V0.1 Visual Audit Report

## Counts

- Selected visual audit cases count: 20
- Frame extraction targets count: 240
- Extracted frames count: 0
- Reused existing frames count: 0
- Missing/unavailable frames count: 240
- Contact sheets generated count: 20

## Planned Extraction Status

- missing_video_source: 240

## Actual Extraction Status

- missing_video_source: 240

## Best Cases For PPT Inspection

These are the first cases to inspect manually. They are not yet PPT-ready visual claims unless frames are later attached and human-verified.

- VA_DEMO_001 / DV01_DEMO_001 / Q1078: Why did Shure laugh when Katrina mentioned her voice messages?
- VA_DEMO_002 / DV01_DEMO_002 / Q617: What decision-making role did Jake take in the group task involving glasses?
- VA_DEMO_003 / DV01_DEMO_003 / Q313: When Katrina discussed her breakup, how did the group mostly respond?
- VA_DEMO_004 / DV01_DEMO_004 / Q1040: Who incorrectly assumed everyone would want an auction group of four items?
- VA_DEMO_005 / DV01_DEMO_005 / Q982: Why didn't Katrina seem fully involved in the conversation about renting costumes during 18:23?

## Blockers

- No local video/frame source was found for the selected MA-EgoQA evidence raw_keys under the searched media roots.
- No reusable frame mapping was found; contact sheets are caption-only placeholders.
- Visual frames alone cannot verify spoken dialogue, speaker identity, or intent; they only support visual plausibility checks.

Searched media roots:

- /scratch/xy3257/ma_egoqa_reproduce

## What Can Be Claimed

- We built a targeted visual audit packet for Dataset V0.1.
- The packet allows human inspection of whether QA/caption evidence is visually plausible.
- Missing visual sources are explicitly marked instead of treated as successful frame extraction.

## What Cannot Be Claimed

- QA labels are final.
- Dataset V0.1 is fully verified.
- Model accuracy improves.
- Historical memory is proven useful.

## Output Files

- `visual_audit_case_subset.csv`
- `visual_frame_extraction_plan.csv`
- `visual_audit_table.csv`
- `visual_audit_packet_gallery.html`
- `visual_audit_report.md`
- `contact_sheets/`
- `frames/`
