# Long-video Understanding Work Snapshot

This repository is a cleaned synchronization snapshot from `/scratch/xy3257` on 2026-06-02.  It contains the active code, validation scripts, and selected experiment outputs for the long-video / egocentric-video understanding work.

## What Is Included

- Root-level AR/VR source-access validation scripts and result summaries.
- `ma_egoqa_reproduce/`: MA-EgoQA historical-memory analysis, dataset construction scripts, and generated dataset/result tables.
- `castle_poc/`: CASTLE proof-of-concept scripts, SLURM jobs, true A/B search outputs, contact sheets, and run logs.
- `castle_hpc/`: remote-access diagnosis, low-bandwidth frame extraction, and event-relevant view-selection validation artifacts.
- `evidence_packets/`: small evidence-packet notes.

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
