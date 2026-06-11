# EgoLife Two-User QA Pilot

This module builds a 20-item pilot set of multiple-choice QA examples from EgoLife videos. Each question is intended to require evidence from at least two users' egocentric streams.

The default model target is `Qwen/Qwen3-VL-8B-Instruct`. No OpenRouter/Gemini API key is used. `HF_TOKEN` is optional and is only used by Hugging Face downloads.

## Pipeline

The main path is now video-first. Qwen3-VL receives the aligned EgoLife videos directly, then a judge/evaluator loop filters out questions that are single-user answerable or not answerable from the combined videos:

```text
EgoLife video + EyeGaze/EyeTracking tree
-> build_manifest
-> prepare_evidence: align at least two users by day/time token and cache videos/gaze
-> generate_video_qa_loop:
   -> generator: directly watches multi-user videos and proposes commonality/difference MCQ
   -> judger: explains why the question was asked and gives feedback
   -> answerability eval: tests single-user videos and combined videos
-> validate_outputs: deterministic schema/gate checks
```

The older `observe_clips -> mine_candidates -> generate_qa -> review_qa` caption/observation path remains available as a legacy/debug baseline, but it is no longer the pilot path.

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

The dry run validates Hugging Face manifest construction, evidence packet preparation, video-first prompt creation, and schema tooling. It does not load Qwen3-VL.

```bash
python -m egolife_two_user_qa build_manifest \
  --days DAY1 \
  --agents A1_JAKE,A2_ALICE \
  --max-per-agent-day 2 \
  --output egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json

python -m egolife_two_user_qa prepare_evidence \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json \
  --output egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --target-count 2 \
  --users-per-case 2 \
  --frames-per-clip 2 \
  --no-download-media

python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20/qa_mcq.video_first.dryrun.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20/video_first_prompts.dryrun.jsonl \
  --target-count 1 \
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
python -m egolife_two_user_qa generate_video_qa_loop \
  --backend openai-compatible-local \
  --base-url http://127.0.0.1:8000/v1 \
  --evidence egolife_two_user_qa/outputs/pilot_20_video_first/evidence_manifest.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20_video_first/video_first_prompts.jsonl \
  --allow-openai-video-input
```

Without `--allow-openai-video-input`, the OpenAI-compatible backend uses sampled frame images as a fallback because not every local server accepts video data URLs.

## Output Schema

Each row in `qa_mcq.jsonl` contains:

- `qa_id`
- `question`
- `options`
- `correct`
- `answer`
- `category`
- `question_type`
- `required_users`
- `evidence`
- `single_user_answerability`
- `combined_answerability`
- `generator_rationale`
- `why_two_users_needed`
- `per_user_evidence_claims`
- `judge_feedback`
- `answerability_eval`
- `attempt_count`
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
