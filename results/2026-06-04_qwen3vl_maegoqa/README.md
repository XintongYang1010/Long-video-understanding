# Qwen3VL MA-EgoQA Fixed-Source Accuracy Results

Date: 2026-06-04
Model: `Qwen/Qwen3-VL-8B-Instruct`
Dataset: official MA-EgoQA JSON with 1741 questions and answer labels.
Evidence: existing 30-second caption BM25 index; no full videos are required for this runner.
Metric: exact multiple-choice accuracy against the official `answer` field.

This is a fixed-source caption/BM25 ablation, not the full MA-EgoQA leaderboard baseline. It keeps the original Gemini scripts unchanged and adds a Qwen-only runner.

## Aggregate Accuracy

| Agent count | Conditions | Mean accuracy |
|---:|---:|---:|
| 1 | 6 | 27.33% |
| 2 | 6 | 31.25% |
| 3 | 2 | 33.86% |

## Condition Results

| Condition | Agents | Items | Correct | Accuracy |
|---|---|---:|---:|---:|
| Lucia | Lucia | 1741 | 510 | 29.29% |
| Tasha | Tasha | 1741 | 443 | 25.45% |
| Shure | Shure | 1741 | 508 | 29.18% |
| Lucia_Tasha | Lucia+Tasha | 1741 | 547 | 31.42% |
| Lucia_Shure | Lucia+Shure | 1741 | 575 | 33.03% |
| Tasha_Shure | Tasha+Shure | 1741 | 545 | 31.30% |
| Jack | Jake | 1741 | 448 | 25.73% |
| Alice | Alice | 1741 | 464 | 26.65% |
| Katrina | Katrina | 1741 | 482 | 27.69% |
| Jack_Alice | Jake+Alice | 1741 | 506 | 29.06% |
| Jack_Katrina | Jake+Katrina | 1741 | 533 | 30.61% |
| Alice_Katrina | Alice+Katrina | 1741 | 558 | 32.05% |
| Jack_Alice_Katrina | Jake+Alice+Katrina | 1741 | 562 | 32.28% |
| Lucia_Tasha_Shure | Lucia+Tasha+Shure | 1741 | 617 | 35.44% |

## Reproduce On NYU Torch

```bash
cd /scratch/$USER/github_sync_long_video_understanding/MA-EgoQA
# Lucia/Tasha/Shure singles and pairs
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1 sbatch hpc/run_qwen3vl_subset_h200.sbatch
# Jack/Alice/Katrina singles and pairs
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1_jack_alice_katrina_v2 QWEN3VL_CONDITIONS="Jack Alice Katrina Jack_Alice Jack_Katrina Alice_Katrina" sbatch hpc/run_qwen3vl_subset_h200.sbatch
# Three-agent combinations
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1_triples QWEN3VL_CONDITIONS="Jack_Alice_Katrina Lucia_Tasha_Shure" sbatch hpc/run_qwen3vl_subset_h200.sbatch
```

Raw predictions are stored as gzip-compressed JSONL files under `predictions/` for per-question win/loss analysis.
