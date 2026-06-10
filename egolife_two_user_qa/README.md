# EgoLife Two-User QA Pilot

This module builds a 20-item pilot set of multiple-choice QA examples from EgoLife videos. Each question is intended to require evidence from at least two users' egocentric streams.

The default model target is `Qwen/Qwen3-VL-8B-Instruct`. No OpenRouter/Gemini API key is used. `HF_TOKEN` is optional and is only used by Hugging Face downloads.

## Pipeline

The main path is now a semantic-complementarity pipeline, not just exact-time multi-user stitching:

```text
EgoLife video + EyeGaze/EyeTracking tree
-> build_manifest
-> observe_clips: Qwen3-VL summarizes each user's clip into structured observations
-> mine_candidates: find user pairs with shared anchors and distinct per-user facts
-> generate_qa: create one MCQ that combines the distinct facts
-> review_qa: Qwen3-VL checks single-user insufficiency and combined-user sufficiency
-> validate_outputs: deterministic schema/gate checks
```

The older `prepare_evidence` command remains as a simple baseline/debug path. For the main pilot, use `observe_clips` plus `mine_candidates`.

## Gaze Projection

EgoLife EyeGaze CSVs are not EgoEverything-style image pixels. They contain Project Aria CPF yaw/pitch/depth values such as `left_yaw_rads_cpf`, `right_yaw_rads_cpf`, `pitch_rads_cpf`, and `depth_m`. The pipeline therefore does not invent `gaze_x/gaze_y`.

By default, gaze summaries are marked:

```json
{"projection_status": "missing_calibration"}
```

To enable 2D gaze points for EgoEverything-style distance/Gaussian sampling, pass an Aria RGB calibration directory:

```bash
python -m egolife_two_user_qa observe_clips \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.json \
  --output egolife_two_user_qa/outputs/pilot_20/observations.jsonl \
  --aria-calibration-dir /path/to/aria_calibrations
```

For strict Aria projection, provide a VRS/no-image VRS file or `online_calibration.jsonl` and install `projectaria-tools`; the code then uses Project Aria's native `CameraCalibration.project()` path. A JSON calibration is also accepted only when it contains explicit RGB intrinsics plus `T_camera_cpf`, or `T_device_camera` and `T_device_cpf`; that JSON route assumes the calibration has already been exported into a pinhole-compatible form. If the public EgoLife Hugging Face files are used as released and no calibration/VRS is supplied, the correct behavior is to keep 2D projection unavailable and rely on video frames plus unprojected 3D gaze statistics.

## Local CPU Dry Run

The dry run validates Hugging Face manifest construction, evidence packet preparation, prompt creation, and schema tooling. It does not load Qwen3-VL.

```bash
python -m egolife_two_user_qa build_manifest \
  --days DAY1 \
  --agents A1_JAKE,A2_ALICE \
  --max-per-agent-day 2 \
  --output egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json

python -m egolife_two_user_qa observe_clips \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json \
  --output egolife_two_user_qa/outputs/pilot_20/observations.dryrun.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20/observation_prompts.dryrun.jsonl \
  --target-clip-count 4 \
  --frames-per-clip 2 \
  --dry-run

python -m egolife_two_user_qa mine_candidates \
  --observations egolife_two_user_qa/outputs/pilot_20/observations.dryrun.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --target-count 1 \
  --min-score 0

python -m egolife_two_user_qa generate_qa \
  --evidence egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20/qa_mcq.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20/generation_prompts.dryrun.jsonl \
  --dry-run
```

## GPU Pilot Run

```bash
bash scripts/run_qwen3vl_gpu.sh \
  --target-count 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 1536
```

For a local OpenAI-compatible server, first start vLLM/SGLang/llama.cpp, then pass:

```bash
python -m egolife_two_user_qa observe_clips \
  --backend openai-compatible-local \
  --base-url http://127.0.0.1:8000/v1 \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.json \
  --output egolife_two_user_qa/outputs/pilot_20/observations.jsonl
```

## Output Schema

Each row in `qa_mcq.jsonl` contains:

- `qa_id`
- `question`
- `options`
- `correct`
- `answer`
- `category`
- `required_users`
- `evidence`
- `single_user_answerability`
- `combined_answerability`
- `review`
- `model_id`
- `source_urls`

Run validation with:

```bash
python -m egolife_two_user_qa validate_outputs \
  --qa egolife_two_user_qa/outputs/pilot_20/qa_mcq.jsonl \
  --csv-output egolife_two_user_qa/outputs/pilot_20/qa_mcq.csv \
  --report egolife_two_user_qa/outputs/pilot_20/generation_report.md \
  --strict-review
```
