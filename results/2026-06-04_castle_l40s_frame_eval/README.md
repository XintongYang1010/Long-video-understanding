# CASTLE/EgoVis Qwen3VL L40S Frame-QA Answer Difference

Date: 2026-06-04
Model: `Qwen/Qwen3-VL-8B-Instruct`
Hardware: one NYU Torch L40S GPU
Official QA source: `EgoVis2026_CVPR_Questions.json` from the CASTLE/EgoVis challenge page

This run uses official question text/options only. The public JSON available locally contains 185 questions but no official answer key, so this directory reports answer difference, not accuracy.

## What Was Evaluated

The manifest mapped already-extracted local CASTLE frames to 2 official questions and 4 frame sets. For each mapped frame set/question pair, the runner asked Qwen3VL with each single source separately and then with a multi-source frame bundle.

## Results

See `summary.md` and `answer_difference.csv`. The clearest answer-change cases are:

- `DAY1_100500000_event`: multi predicted `D`, while all 4 single-source predictions differed (`A:1; C:3`).
- `EX6_HPC_Q5_TABLETOP`: multi predicted `B`, while all 5 single-source predictions were `A`.

These are evidence that multi-view context changes model decisions. They are not evidence that the multi-view answer is more accurate until the official answer key or Codabench score is available.

## Reproduce On NYU Torch

```bash
cd /scratch/$USER/github_sync_long_video_understanding
QWEN3VL_LIMIT=none sbatch hpc/run_qwen3vl_castle_frames_l40s.sbatch
```

The L40S job uses OOM guards from the successful run: one frame per source, at most 8 multi-source frames total, and `max_image_pixels=262144`.
