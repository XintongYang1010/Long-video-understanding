# Dataset V0.1 Visual Audit With Frames Report

## FFmpeg Status

- ffmpeg path: /scratch/xy3257/castle_hpc/envs/castle/bin/ffmpeg
- ffmpeg version: ffmpeg version 8.0.1 Copyright (c) 2000-2025 the FFmpeg developers
- ffprobe version: ffprobe version 8.0.1 Copyright (c) 2007-2025 the FFmpeg developers

## Counts

- Unique videos requested: 80
- Unique videos downloaded or cached: 80
- Frames planned: 240
- Frames extracted: 240
- Frames missing: 0
- Contact sheets generated: 20

## Download Status

- cached: 80

## Extraction Status

- already_extracted: 238
- already_extracted_clamped_to_clip_end: 2

## Top 5 PPT-Ready Cases With Frames

- VA_DEMO_001 / DV01_DEMO_001 / Q1078 (12 frames): Why did Shure laugh when Katrina mentioned her voice messages?
- VA_DEMO_002 / DV01_DEMO_002 / Q617 (12 frames): What decision-making role did Jake take in the group task involving glasses?
- VA_DEMO_003 / DV01_DEMO_003 / Q313 (12 frames): When Katrina discussed her breakup, how did the group mostly respond?
- VA_DEMO_004 / DV01_DEMO_004 / Q1040 (12 frames): Who incorrectly assumed everyone would want an auction group of four items?
- VA_DEMO_005 / DV01_DEMO_005 / Q982 (12 frames): Why didn't Katrina seem fully involved in the conversation about renting costumes during 18:23?

## Blockers

- 2 planned frame times exceeded the source clip duration and were clamped to the clip end.
- Visual frames alone cannot verify dialogue, speaker identity, or intent.

## Claim Boundary

Can say:
- We built targeted visual audit packets for a small Dataset V0.1 subset.
- Frames are used only for sanity-checking caption/evidence plausibility.

Cannot say:
- QA labels are final.
- Dataset is fully verified.
- Historical memory is proven useful.
- Model accuracy improves.
