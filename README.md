# Long-video Understanding Work Snapshot

This repository is a cleaned synchronization snapshot from `/scratch/xy3257` on 2026-06-02.  It contains the active code, validation scripts, and selected experiment outputs for the long-video / egocentric-video understanding work.


## 2026-06-04 Qwen3VL Progress

This branch adds two Qwen3VL experiment tracks that were run on NYU Torch with account `torch_pr_674_tandon_advanced`.

### MA-EgoQA Fixed-Source Accuracy

- Model: `Qwen/Qwen3-VL-8B-Instruct`.
- Runner: `MA-EgoQA/egomas/src/inference_qwen3vl_subset.py`.
- Evidence: official MA-EgoQA questions plus the existing 30-second caption BM25 index; full videos are not required for this runner.
- Metric: official answer accuracy over all 1741 MA-EgoQA questions.
- Original Gemini inference code is unchanged; this is a Qwen-only fixed-source ablation.

| Condition | Agents | Accuracy |
|---|---|---:|
| Jack | Jake | 25.73% |
| Alice | Alice | 26.65% |
| Katrina | Katrina | 27.69% |
| Jack_Alice | Jake+Alice | 29.06% |
| Jack_Katrina | Jake+Katrina | 30.61% |
| Alice_Katrina | Alice+Katrina | 32.05% |
| Lucia | Lucia | 29.29% |
| Tasha | Tasha | 25.45% |
| Shure | Shure | 29.18% |
| Lucia_Tasha | Lucia+Tasha | 31.42% |
| Lucia_Shure | Lucia+Shure | 33.03% |
| Tasha_Shure | Tasha+Shure | 31.30% |
| Jack_Alice_Katrina | Jake+Alice+Katrina | 32.28% |
| Lucia_Tasha_Shure | Lucia+Tasha+Shure | 35.44% |

Across these fixed-source ablations, single-agent mean accuracy is about 27.33%, pair-agent mean accuracy is about 31.41%, and three-agent mean accuracy is 33.86%. This supports a preliminary source-aggregation gain, but the gain is modest and source-dependent.

Artifacts: `results/2026-06-04_qwen3vl_maegoqa/` contains summary tables and gzip-compressed per-question predictions for follow-up win/loss analysis.

### CASTLE/EgoVis Frame-QA Answer Difference

- Model: `Qwen/Qwen3-VL-8B-Instruct`.
- Runner: `castle_qwen_eval/run_qwen3vl_frame_qa.py`.
- Hardware: one L40S GPU via `hpc/run_qwen3vl_castle_frames_l40s.sbatch`.
- Official QA: `EgoVis2026_CVPR_Questions.json` contains question/options only; no local answer key is available.
- Metric: answer difference between single-source and multi-source frame prompts, not accuracy.

The manifest mapped existing extracted CASTLE frames to 2 official questions and 4 frame sets. The strongest answer-change cases were `DAY1_100500000_event` and `EX6_HPC_Q5_TABLETOP`, where every single-source prediction differed from the multi-source prediction. This shows that multi-view context changes Qwen decisions, but it does not prove accuracy improvement until an official answer key or Codabench score is available.

Artifacts: `results/2026-06-04_castle_l40s_frame_eval/` contains `summary.md`, `answer_difference.csv`, `frame_manifest.json`, `submission_partial.json`, and the small `predictions.jsonl`.

### Reproduce

```bash
# Shared Qwen environment helper for Slurm jobs.
source hpc/env_qwen3vl.sh

# MA-EgoQA Lucia/Tasha/Shure singles and pairs on H200.
cd /scratch/$USER/github_sync_long_video_understanding/MA-EgoQA
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1 sbatch hpc/run_qwen3vl_subset_h200.sbatch

# MA-EgoQA Jack/Alice/Katrina singles and pairs.
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1_jack_alice_katrina_v2 QWEN3VL_CONDITIONS="Jack Alice Katrina Jack_Alice Jack_Katrina Alice_Katrina" sbatch hpc/run_qwen3vl_subset_h200.sbatch

# MA-EgoQA triples.
QWEN3VL_LIMIT=none QWEN3VL_OUTPUT_ROOT=outputs/qwen3vl_subset_full_gpu1_triples QWEN3VL_CONDITIONS="Jack_Alice_Katrina Lucia_Tasha_Shure" sbatch hpc/run_qwen3vl_subset_h200.sbatch

# CASTLE/EgoVis mapped frame QA on L40S.
cd /scratch/$USER/github_sync_long_video_understanding
QWEN3VL_LIMIT=none sbatch hpc/run_qwen3vl_castle_frames_l40s.sbatch
```

The manual historical-audit CSVs in `ma_egoqa_reproduce/outputs/` are still audit scaffolding, not a completed labeled dataset. Treat them as candidate-analysis material until human labels are filled.

## What Is Included

- Root-level AR/VR source-access validation scripts and result summaries.
- `ma_egoqa_reproduce/`: MA-EgoQA historical-memory analysis, dataset construction scripts, and generated dataset/result tables.
- `castle_poc/`: CASTLE proof-of-concept scripts, SLURM jobs, true A/B search outputs, contact sheets, and run logs.
- `castle_hpc/`: remote-access diagnosis, low-bandwidth frame extraction, and event-relevant view-selection validation artifacts.
- `evidence_packets/`: small evidence-packet notes.
- `castle_qwen_eval/`: Qwen3VL CASTLE/EgoVis official-question frame evaluation tools.
- `results/2026-06-04_*`: curated Qwen3VL MA-EgoQA and CASTLE experiment artifacts.

The nested `ma_egoqa_reproduce/MA-EgoQA/` directory is an upstream reference checkout from `https://github.com/KangsanKim07/MA-EgoQA.git`; its nested `.git` directory is not included here.

## Important Files

Root-level validation and summary files:

- `build_arvr_source_access_packet.py`
- `validate_source_access_seed_cases.py`
- `validate_audio_speech_cases.py`
- `source_access_seed_cases_v0_3.csv` / `.md`
- `source_access_validation_v0_3.csv` / `.md`
- `audio_validation_v0_3.csv`
- `audio_validation_snippets.md`
- `cases_needing_frame_extraction.csv`
- `cases_to_reject_or_weaken.csv`
- `cases_verified_by_transcript.csv`
- `visual_audit_summary.md`
- `arvr_source_access_validation_bundle.zip`
- `Dataset_V0_2_Verified_20260529.tar.gz`

MA-EgoQA results:

- `ma_egoqa_reproduce/build_dataset_v0_*.py`
- `ma_egoqa_reproduce/build_evidence_scope_*.py`
- `ma_egoqa_reproduce/maegoqa_historical_feasibility_v1.py`
- `ma_egoqa_reproduce/outputs/historical_v2_fullscreen/dataset_v0_2_verified/`
- `ma_egoqa_reproduce/outputs/historical_v2_fullscreen/dataset_v0_1/`
- `ma_egoqa_reproduce/outputs/historical_v2_fullscreen/evidence_scope_fullscreen_v2_report.md`
- `ma_egoqa_reproduce/outputs/historical_v2_fullscreen/tier_*_cases_v2.csv`
- `MA-EgoQA/egomas/src/inference_qwen3vl_subset.py`
- `MA-EgoQA/egomas/src/summarize_qwen3vl_subset.py`
- `MA-EgoQA/hpc/run_qwen3vl_subset_h200.sbatch`
- `results/2026-06-04_qwen3vl_maegoqa/`

CASTLE results:

- `castle_poc/castle_incremental_qa_poc.py`
- `castle_poc/castle_extract_true_AB_candidates.py`
- `castle_poc/castle_extract_true_AB_candidates_round2.py`
- `castle_poc/true_AB_search_summary.md`
- `castle_poc/true_AB_search_summary_round2.md`
- `castle_poc/candidate_true_AB_cases*.csv`
- `castle_poc/candidate_true_AB_contact_sheets*/`
- `castle_poc/castle_true_ab_round2_results_minimal.tar.gz`
- `castle_poc/results/demo_v1/`
- `castle_hpc/castle_remote_access_diagnosis.py`
- `castle_hpc/castle_low_bandwidth_remote_frame_test.py`
- `castle_hpc/castle_event_relevant_view_selection.py`
- `castle_qwen_eval/prepare_castle_official_frame_manifest.py`
- `castle_qwen_eval/run_qwen3vl_frame_qa.py`
- `castle_qwen_eval/summarize_frame_qa.py`
- `hpc/run_qwen3vl_castle_frames_l40s.sbatch`
- `results/2026-06-04_castle_l40s_frame_eval/`

## Environment Configuration

The original working root was:

```bash
/scratch/xy3257
```

The environment directories themselves are intentionally not committed.  They remain on the original machine, but this repository records the installed configuration for collaborators.

Detected conda environments:

```text
base                                      /scratch/xy3257/miniforge3
castle HPC env                            /scratch/xy3257/castle_hpc/envs/castle
CASTLE POC conda env                      /scratch/xy3257/castle_poc/cenv
MA-EgoQA conda env                        /scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa
```

Base Miniforge:

```text
path:   /scratch/xy3257/miniforge3
python: 3.13.13
conda:  26.3.2
mamba:  2.6.0
pip:    26.0.1
```

Recommended CASTLE POC environment:

```text
path:            /scratch/xy3257/castle_poc/cenv
python:          3.10.20
ffmpeg:          6.1.2
huggingface_hub: 1.14.0
numpy:           2.2.6
pandas:          2.3.3
pillow:          12.2.0
requests:        2.33.1
tqdm:            4.67.3
```

CASTLE HPC environment:

```text
path:     /scratch/xy3257/castle_hpc/envs/castle
python:   3.10.20
ffmpeg:   8.0.1
numpy:    2.2.6
pandas:   2.3.3
pillow:   12.2.0
requests: 2.33.1
tqdm:     4.67.3
```

MA-EgoQA environment:

```text
path:     /scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa
python:   3.10.20
numpy:    2.2.6
pandas:   2.3.3
requests: 2.34.2
tqdm:     4.67.3
```

Additional Python venvs found under `castle_poc/`:

```text
castle_poc/venv
  python:          3.12.12
  numpy:           2.4.4
  pandas:          3.0.2
  pillow:          12.2.0
  requests:        2.33.1
  tqdm:            4.67.3
  huggingface_hub: 1.14.0

castle_poc/venv_compute
  metadata python version: 3.9.21
  current status: not recommended; python failed to start because libpython3.9.so.1.0 was not found.
```

On the original machine, activate environments by prefix:

```bash
source /scratch/xy3257/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/xy3257/castle_poc/cenv
conda activate /scratch/xy3257/castle_hpc/envs/castle
conda activate /scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa
```

For a fresh machine, recreate with Python 3.10 and install the package versions above.  Some scripts also require dataset/video access paths from the original HPC or Hugging Face storage.

## Cleanup And Upload Policy

Before upload, only cache-like artifacts were deleted from the original workspace:

```text
__pycache__/
*.pyc
hf_cache/
video_cache/
```

The following were not uploaded:

```text
.codex/
.huggingface/
.vscode-server/
.local/
miniforge3/
npm-global/
venv/
venv_compute/
cenv/
envs/
hf_cache/
video_cache/
__pycache__/
*.pyc
debug *.bin probes
Miniforge3.sh
```

This keeps the repository focused on code and interpretable experiment outputs while preserving local login/session state outside git.

## Notes For Collaborators

- Some scripts contain absolute paths or assumptions tied to `/scratch/xy3257`; adjust paths before rerunning elsewhere.
- Large CSV outputs are included because they are the primary experimental artifacts.  No uploaded file is over GitHub's 100MB hard limit.
- Contact sheets and selected extracted JPG evidence frames are included for qualitative audit and visual validation.  Large video caches and Hugging Face download caches are not included.
