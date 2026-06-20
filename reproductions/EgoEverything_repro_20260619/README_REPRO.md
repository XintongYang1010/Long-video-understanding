# EgoEverything Reproduction Notes

Workspace on Torch:

```bash
BASE=/scratch/xy3257/EgoEverything_repro_20260619
REPO=$BASE/EgoEverything
```

Repository:

- Upstream: https://github.com/ElecBeholder/EgoEverything
- Checked commit: `41d91cf529b2a0a889d5ea41e0b488c50106443d`
- Branch: `fix`

Paper target:

- https://arxiv.org/html/2604.08342v1#S4
- Section 4 evaluates EgoEverything on FR, AD, GC, GM, VMP, and AMEGO settings.
- The released repository is primarily the data processing and MCQ generation pipeline, not a complete packaged Section 4 model-evaluation runner.

Verified setup:

- Python virtual environment: `$BASE/.venv`
- Python version: 3.12
- `pip check`: no broken requirements
- Cached model weights:
  - `$BASE/home/.cache/clip/ViT-L-14.pt`
  - `$BASE/cache/torch/hub/checkpoints/resnet50-0676ba61.pth`
- Smoke test on Slurm `cpu_short` completed successfully with an empty dataset JSON.
- AEA one-sequence download/process job `11109270` completed successfully using fresh Project Aria Dataset Explorer links.
- Gaze/tracking visualization job `11109709` completed successfully using the repo's `gaze_hand_visualizer.py`.
- Safety patch check smoke job `11109989` completed successfully after removing hardcoded OpenRouter defaults from the local checkout.
- Section 4 condition-preprocessing job `11110469` completed successfully for FR, AD, GC, and GM on the verified AEA sequence using the arXiv-source `1/3` AD/GC ratio.
- Repro audit job `11112793` completed successfully to validate the Section 4 condition videos, re-check public release assets, and refresh the strict artifact-readiness report.

Important environment detail:

The login node and compute nodes expose different `/usr/bin/python3` targets. The virtualenv was adjusted so `$BASE/.venv/bin/python3` points to `/usr/bin/python3.12`, which exists on both sides.

Always source:

```bash
source /scratch/xy3257/EgoEverything_repro_20260619/env.sh
```

Before running commands that call Gemini/OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

Expected input layout after data preparation:

```text
$DATASET_PATH/
  sequence_id/
    sequence_id.mp4
    sequence_id_summary.json
    sequence_id_tracking.csv

$DATASET_JSON
```

Default paths:

```bash
DATASET_PATH=$BASE/Data
DATASET_JSON=$BASE/Data/AriaEverydayActivities_download_urls.json
DATASET_NAME=AriaEveryday_Activities
```

Verified one-sequence data:

```text
$DATASET_PATH/loc5_script4_seq6_rec1/loc5_script4_seq6_rec1.mp4
$DATASET_PATH/loc5_script4_seq6_rec1/loc5_script4_seq6_rec1_tracking.csv
$DATASET_PATH/loc5_script4_seq6_rec1/loc5_script4_seq6_rec1_with_tracking.mp4
```

- Video: 4263 frames, 20 FPS, 1408x1408.
- Tracking CSV: 4263 rows, gaze coordinates present for all rows.
- Visualization video: 4263 frames, 20 FPS, 1408x1408.
- Manifest source: `https://explorer.projectaria.com:443/data/aea/download_links`.
- AEA annotations were also downloaded. For this sequence they contain `metadata.json` and a sparse `speech.csv`; they are not a substitute for the action summary JSON expected by the VQA generation pipeline.

Run order for a one-sequence end-to-end check:

```bash
source $BASE/env.sh

# Uses the current Project Aria Dataset Explorer AEA download-links endpoint.
REFRESH_AEA_MANIFEST=1 sbatch $BASE/scripts/run_download_aea_one.sbatch

# Requires OPENROUTER_API_KEY.
bash $BASE/scripts/submit_one_sequence_pipeline.sh
```

This submits `run_video_summarizer_one.sbatch` first, then submits `run_vqa_one.sbatch` with an `afterok` dependency. If a sequence summary already exists, it submits only VQA generation.

The helper was tested without `OPENROUTER_API_KEY`; it failed fast and submitted no Slurm jobs, as intended.

Local safety patch:

The upstream checkout under `$REPO` intentionally differs from `origin/fix` in three files:

```text
Generation/example_config.sh
ai_video_agent.py
video_summarizer.py
```

These local changes remove hardcoded OpenRouter-style default keys from the reproduction checkout and require `OPENROUTER_API_KEY` to be set explicitly. `grep -R "sk-or-v1-" $REPO` returns no matches after this patch.

Run order for the verified empty smoke test:

```bash
sbatch $BASE/scripts/run_smoke_empty.sbatch
```

Run order for the verified gaze/tracking visualization:

```bash
sbatch $BASE/scripts/run_gaze_visualizer_one.sbatch
```

Run the Section 4 artifact readiness checker:

```bash
python $BASE/scripts/check_section4_artifacts.py \
  --artifact-root $BASE/section4_artifacts \
  --dataset-path $DATASET_PATH \
  --repo $EGOEVERYTHING_REPO \
  --report $BASE/section4_artifact_check.json
```

The checker is intentionally strict. It returns nonzero until the official MCQ benchmark manifest, answer key/gold labels, Section 4 evaluation runner/prompts, processed data, and model/API credentials are all present.

Run the Section 4 FR/AD/GC/GM video-condition preprocessing:

```bash
sbatch $BASE/scripts/run_section4_preprocess_one.sbatch
```

Verified outputs are under:

```text
$BASE/section4_conditions/loc5_script4_seq6_rec1/
  FR/loc5_script4_seq6_rec1.mp4
  AD/loc5_script4_seq6_rec1.mp4
  GC/loc5_script4_seq6_rec1.mp4
  GM/loc5_script4_seq6_rec1.mp4
  section4_conditions_manifest.json
```

The default `GC_CROP_RATIO=1/3` and `AD_SCALE=1/3` follow the arXiv source text: AD resizes to `1/3 H x 1/3 W`, and GC uses a gaze-centered square crop with the same per-frame token budget. The camera-ready prose describes these as roughly `10%` of the original frame/resolution. Output dimensions are rounded to the nearest codec-safe even integer so MP4 writers preserve the manifest dimensions.

Run the reproducibility audit wrapper:

```bash
sbatch $BASE/scripts/run_repro_audit.sbatch
```

This writes:

```text
$BASE/section4_conditions_validation.json
$BASE/public_release_check.json
$BASE/section4_artifact_check.json
$BASE/repro_audit_summary.json
```

Latest audit result (`11112793`):

```text
condition_validation.valid: true
public_release_check.public_section4_assets_ready: false
section4_artifact_check.ready: false
section4_artifact_check missing: official Section 4 MCQ benchmark manifest; official answer key / gold labels; Section 4 VLM evaluation runner/prompts for FR, AD, GC, GM, VMP, AMEGO; model/API credential environment variable
```

Known remaining external requirements:

- A valid OpenRouter API key.
- For full Section 4 replication, a released MCQ benchmark/evaluation manifest is still required; the current upstream repo does not include the final 5,000+ MCQ evaluation set or a complete VLM benchmark runner.
