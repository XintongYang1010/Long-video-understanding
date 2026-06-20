# Multi-user EgoLife QA Handoff

Audience: Jiajing / Xuerong
Last local scan: 2026-06-21
Primary repo: `Long-video-understanding`

This document is intentionally candid. It separates the code that is already on `origin/main` from local ignored outputs and unmerged experimental branches.

## 1. Project Goal

The project is trying to build a multi-user egocentric QA benchmark from EgoLife videos. The key idea is not ordinary video QA generation. We want questions that cannot be answered from one user's first-person video alone, and that require another user's, or multiple users', visual/time/context evidence.

A good target question should feel like a realistic AR assistant or memory question. It should start from what one user could naturally know from their own view, then ask for a missing detail that is only visible or disambiguated in another user's view.

Examples of the intended dependency:

- User A sees themselves leaving, carrying, pointing, preparing, or interacting with something.
- User B's view reveals what happened next, who handled an object, how someone reacted, or what state the room/object ended in.
- If User B's video is removed, the answer should become unknowable or ambiguous.

## 2. Current Status

Completed:

- The `egolife_two_user_qa` package has a working video-first pipeline on `origin/main`.
- The main path can build an EgoLife manifest, prepare two-user evidence packets, call a VLM generator, run a VLM judger, run answerability checks, and validate outputs.
- The current deterministic/unit tests pass locally with `python3 -m unittest egolife_two_user_qa.tests.test_core`.
- Prompt, judge, schema, answerability, and human-review trace infrastructure exist.
- EgoEverything reproduction support has been merged into `origin/main` under `reproductions/EgoEverything_repro_20260619/`.
- MA-EgoQA / EgoMAS / Route B code exists as baseline and background context, but it is not the current two-user QA generator.

In progress:

- The strongest recent prompt work appears to be on the unmerged branch `codex/egolife-qa-32b-relaxed-human-review`.
- That branch adds relaxed natural prompts, richer taxonomy fields, physical identity checks, speaker entitlement rules, screen/privacy constraints, time-stratified evidence sampling, balanced pair selection, Gemini replay support, and an answerability-skip mode for human-review candidate pools.
- Several local review bundles exist under `egolife_two_user_qa/outputs/review_*`, but these are ignored by git and are not available to someone who only clones GitHub.

Not finished:

- There is no final clean benchmark set yet.
- Human review labels have not been consolidated into a tracked artifact.
- The latest relaxed/human-review branch is not merged into `origin/main`.
- The local prompt examples and review bundles are not organized into a tracked handoff package.
- Baseline/evaluation against a final generated QA set has not been completed.
- Some notebooks and README snippets still contain stale clone URLs or older branch names.

Prototype only:

- `observe_clips` and `mine_candidates` are useful debugging/prototyping tools, but the current README explicitly says they are not the formal pilot path.
- The 32B relaxed human-review candidate pools should not be treated as final benchmark quality.

Reusable:

- Manifest parsing, evidence packet construction, video-first prompt loop, judge prompt, answerability gate, strict schema validation, and human review sheet generation are reusable.
- Local failure analyses are useful for prompt iteration, but should be curated before being shared as formal examples.

## 3. Repository Structure

Main two-user QA package:

- `egolife_two_user_qa/README.md`
  Current package-level overview and run instructions for the video-first pilot.
- `egolife_two_user_qa/cli.py`
  CLI entrypoint for `build_manifest`, `prepare_evidence`, `observe_clips`, `mine_candidates`, `generate_video_qa_loop`, and `validate_outputs`.
- `egolife_two_user_qa/manifest.py`
  Reads EgoLife video and gaze dataset trees, parses filenames, and aligns video/gaze metadata by day, agent, and time token.
- `egolife_two_user_qa/evidence.py`
  Builds evidence packets from aligned clips. Current `origin/main` groups by day/time token and selects required users from clips available at the same time token.
- `egolife_two_user_qa/prompts.py`
  Main generator prompt, judger prompt, answerability prompt, output schema text, and prompt design rules.
- `egolife_two_user_qa/video_qa_loop.py`
  End-to-end generation loop: choose question type, call generator, parse JSON, run judger, run answerability checks, write accepted/rejected/intermediate outputs.
- `egolife_two_user_qa/schema.py`
  Required fields, review gates, strict validation, CSV export, and human review sheet formatting.
- `egolife_two_user_qa/qa_pipeline.py`
  Output validation/report helper.
- `egolife_two_user_qa/qwen3vl_runner.py`
  VLM runner abstraction. Current main supports `transformers-local`, `openai-compatible-local`, and `dry-run`.
- `egolife_two_user_qa/observations.py`
  Older per-clip observation summarization path. Useful for debugging, not the formal current pilot path.
- `egolife_two_user_qa/candidate_mining.py`
  Older semantic candidate mining from observations. Useful for experiments, not the formal current pilot path.
- `egolife_two_user_qa/JUDGE_DESIGN_NOTES.md`
  Important design notes for the judge dimensions and answerability gate.
- `egolife_two_user_qa/tests/test_core.py`
  Unit tests for manifest parsing, gaze behavior, prompt/schema behavior, gates, and dry-run provenance.

Run scripts:

- `scripts/run_qwen3vl_gpu.sh`
  Main local/Torch shell script for current video-first pilot runs.
- `hpc/run_egolife_two_user_qwen3vl.sbatch`
  Slurm wrapper for Torch. Contains account, environment, cache, and output assumptions.
- `hpc/env_qwen3vl.sh`
  Torch environment helper for Qwen/VLM jobs.

Related background:

- `MA-EgoQA/`
  MA-EgoQA / EgoMAS baseline code and Qwen3-VL / Route B experiments.
- `maegoqa_schema_summary.md`
  Useful schema and category summary for MA-EgoQA.
- `maegoqa_vs_egovis_dataset_choice.md`
  Dataset-choice notes and claim boundaries.
- `reproductions/EgoEverything_repro_20260619/`
  EgoEverything reproduction support package. Useful reference, not the current EgoLife two-user QA generation path.

Local-only / ignored:

- `egolife_two_user_qa/outputs/review_*`
  Local review bundles and failure analyses. These are ignored by git.
- `outputs/egolife_prompt_examples/`
  Local prompt example assets. These are also ignored/local-only.
- Top-level `/Users/xintongyang/Documents/NYU` contains extra local folders and a separate no-remote git repo. The real GitHub project is the nested `Long-video-understanding` repo.

## 4. End-to-End Pipeline

### Step 0: data preparation

Code:

- `egolife_two_user_qa/manifest.py`
- CLI command: `build_manifest`

Input:

- EgoLife video dataset: `lmms-lab/EgoLife`
- EgoLife gaze dataset: `Wangtwohappy/EgoLife_EyeTracking_EyeGaze`
- Optional day/agent filters.

Output:

- `manifest.json`, usually under an output directory such as `egolife_two_user_qa/outputs/pilot_20_video_first/`.

How to run:

```bash
python -m egolife_two_user_qa build_manifest \
  --output egolife_two_user_qa/outputs/pilot_20_video_first/manifest.json
```

Notes:

- On this local Mac shell, `python` was not available, but `python3` was. On Torch/conda, `python` is likely correct after activating the env.
- Hugging Face access may need `HF_TOKEN` for rate limits or private access, but current README says the QA pipeline does not use OpenRouter/Gemini API keys on main.

### Step 1: candidate clip / user pair selection

Code:

- `egolife_two_user_qa/evidence.py`
- CLI command: `prepare_evidence`

Input:

- `manifest.json`
- Optional cache directory.

Intermediate data structure:

- Evidence packets with fields like `evidence_id`, `day`, `time_token`, `clip_clock`, `required_users`, `requirement`, `clips`, and `source_urls`.

Output:

- `evidence_manifest.jsonl`
- Cached videos/gaze/frames if media download is enabled.

How to run:

```bash
python -m egolife_two_user_qa prepare_evidence \
  --manifest egolife_two_user_qa/outputs/pilot_20_video_first/manifest.json \
  --output egolife_two_user_qa/outputs/pilot_20_video_first/evidence_manifest.jsonl \
  --target-count 80 \
  --users-per-case 2 \
  --frames-per-clip 4
```

Notes:

- Current `origin/main` selection is mostly same-time-token grouping. It does not guarantee semantic complementarity by itself.
- The unmerged relaxed/human-review branch adds more sampling controls such as time stratification and balanced pair selection.

### Step 2: prompt construction

Code:

- `egolife_two_user_qa/prompts.py`
- Functions: `build_video_generation_prompt`, `build_judger_prompt`, `build_answerability_prompt`

Input:

- One evidence packet.
- Target question type.
- Optional feedback from previous failed attempts.

Output:

- Generator prompt.
- Judger prompt.
- Answerability prompt.

Notes:

- Current main uses raw videos when supported. For OpenAI-compatible backends without video support, the code can fall back to sampled frames.
- Generator prompt is currently compact and focused on `commonality` / `difference`.
- The more elaborate natural-language/taxonomy prompt is on `codex/egolife-qa-32b-relaxed-human-review`, not on `origin/main`.

### Step 3: LLM QA generation

Code:

- `egolife_two_user_qa/video_qa_loop.py`
- `egolife_two_user_qa/qwen3vl_runner.py`
- CLI command: `generate_video_qa_loop`

Input:

- `evidence_manifest.jsonl`
- Video files or sampled frames.
- Model/backend config.

Output:

- Candidate QA JSON.
- Raw model outputs.
- Intermediate generation trace.

How to run through the script:

```bash
bash scripts/run_qwen3vl_gpu.sh \
  --target-count 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 1536
```

Notes:

- Default main model is `Qwen/Qwen3-VL-8B-Instruct`.
- The run script defaults to `transformers-local`.
- TODO: exact command for reproducing the 32B relaxed human-review pools should be taken from the unmerged branch and Slurm logs, not reconstructed from memory.

### Step 4: judging / filtering

Code:

- `egolife_two_user_qa/prompts.py`
- `egolife_two_user_qa/video_qa_loop.py`

Input:

- Candidate QA.
- Same evidence videos/frames.

Output:

- Judger JSON.
- Gate pass/fail status.
- Feedback for retry.

Blocking judge dimensions on current main:

- `first_person_naturalness`
- `agent_perspective`
- `source_scope`
- `question_type_semantics`
- `multi_video_necessity`
- `visual_grounding`
- `mcq_option_quality`
- `gaze_safety`
- `human_auditability`

Notes:

- The judge must reject questions that are just timestamp overlap, generic comparison, or answerable from one user's view.
- The judge is necessary but not sufficient. Human review still catches many issues.

### Step 5: saving outputs

Code:

- `egolife_two_user_qa/video_qa_loop.py`
- `egolife_two_user_qa/schema.py`
- `egolife_two_user_qa/qa_pipeline.py`

Output files usually include:

- `qa_mcq.jsonl`
- `qa_mcq.csv`
- `human_review_sheet.md`
- `generation_report.md`
- `video_first_prompts.jsonl`
- `qa_mcq.intermediate.jsonl`
- `qa_mcq.rejected.jsonl`

Notes:

- Only a small README under `egolife_two_user_qa/outputs/pilot_20/` is tracked.
- Most real run outputs are ignored and local-only.

### Step 6: human review

Code/output:

- `human_review_sheet.md`
- `accepted_review.md`
- `accepted_review.html`
- `summary.json`
- `accepted_json/`
- `prompts/`
- `raw/`

What to inspect:

- Does the question sound like the speaker could naturally ask it?
- Is there a clear self/speaker anchor in one user's video?
- Is the answer actually visible or inferable from the other user's video?
- Would removing any required user make the question unanswerable or ambiguous?
- Are answer options mutually exclusive and plausible?
- Does the answer align with the cited video evidence?
- Are timestamps, user IDs, and visible people aligned?
- Is the question avoiding private screen/message content unless genuinely visible and appropriate?

Notes:

- Some review bundles omit `manifest.json` and `evidence_assets/` to keep them small. The complete source output may only exist on Torch scratch.

### Step 7: next possible baseline / evaluation

Likely next steps:

- Select a small human-approved set from the candidate pools.
- Freeze the prompt/schema branch used to generate it.
- Commit or archive the exact prompts, evidence IDs, and review decisions.
- Run self-only, other-only, pair/all-user baselines.
- Compare whether multi-user evidence gives a measurable answerability or accuracy lift.

TODO:

- Define final benchmark split and exact evaluation protocol.
- Decide whether the final generator should use main's strict answerability gate or the relaxed/human-review branch with manual filtering.

## 5. How to Run

Environment setup:

```bash
cd /Users/xintongyang/Documents/NYU/Long-video-understanding

# Local sanity test on this machine:
python3 -m unittest egolife_two_user_qa.tests.test_core
```

Required packages:

- `requirements-egolife-two-user-qa.txt`
- `requirements-qwen3vl.txt`
- CUDA PyTorch must be installed separately for Torch/GPU runs.
- `ffmpeg` / `ffprobe` are needed for video probing and frame extraction.

Torch environment:

```bash
source hpc/env_qwen3vl.sh
```

Data path:

- Current code can download/cache from Hugging Face.
- Torch scripts assume paths like `/scratch/${USER}/github_sync_long_video_understanding` and cache/output dirs under `/scratch/${USER}/`.
- TODO: verify the collaborator's Torch username and scratch paths before running long jobs.

API/model config:

- Current main default: `Qwen/Qwen3-VL-8B-Instruct`.
- Current main does not require commercial API keys for local Qwen inference.
- `HF_TOKEN` may help with Hugging Face download/rate limits.
- `openai-compatible-local` backend uses `LOCAL_VLM_API_KEY` or a placeholder.
- MA-EgoQA original Gemini/EgoMAS code and the unmerged Gemini replay branch need `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Generation command:

```bash
bash scripts/run_qwen3vl_gpu.sh \
  --target-count 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 1536
```

Judge command:

- There is no separate main judge command. Judging is inside `generate_video_qa_loop`.

Validation command:

```bash
python -m egolife_two_user_qa validate_outputs \
  --qa egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.jsonl \
  --csv-output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.csv \
  --report egolife_two_user_qa/outputs/pilot_20_video_first/generation_report.md \
  --strict-review
```

Output location:

- Default script output: `egolife_two_user_qa/outputs/pilot_20_video_first`
- Existing local review bundles: `egolife_two_user_qa/outputs/review_*`
- Important: these review bundles are ignored by git unless explicitly copied into a tracked folder.

## 6. Prompt Files and Design

Generator prompt:

- Current main: `egolife_two_user_qa/prompts.py`, function `build_video_generation_prompt`.

Judger prompt:

- Current main: `egolife_two_user_qa/prompts.py`, function `build_judger_prompt`.

Answerability prompt:

- Current main: `egolife_two_user_qa/prompts.py`, function `build_answerability_prompt`.

Few-shot examples:

- Current main has an embedded good example in the generator prompt.
- Additional selected prompt examples exist locally under `outputs/egolife_prompt_examples/`, but they are ignored and not tracked.

Core prompt rules on current main:

- Use raw video evidence, not captions or imagined observations.
- Generate exactly one 5-option multiple-choice QA.
- Start from a speaker-side anchor in one user's view.
- Ask for a missing detail that requires another user's view.
- Any single required user alone should be insufficient.
- Combined required users should be sufficient.
- Avoid generic "what was the other person doing nearby" questions.
- Avoid timestamp-only or superficial commonality/difference questions.

Known prompt problems:

- Main prompt is stricter and cleaner, but candidate selection is weak, so the model can still produce timestamp-stitching or generic comparison.
- Earlier prompt versions produced template collapse.
- Later relaxed prompts improved naturalness but introduced quality variability and required more human review.
- Prompt versions are spread across main, unmerged branches, ignored output snapshots, HTML renderings, and notebooks.

Recommended prompt direction:

- Keep the speaker/self anchor requirement.
- Keep explicit other-view-only target tests.
- Keep physical identity and same-event binding checks from the relaxed branch.
- Keep screen/privacy restrictions.
- Reduce overlong rule lists where they cause stiff or repetitive questions.
- Add curated few-shot examples from actual accepted/rejected local outputs, but only after manual review.

## 7. QA Quality Criteria

A good multi-user EgoLife QA should satisfy:

- It has a clear speaker/self anchor: "I was doing/seeing X".
- The speaker anchor is actually supported by the speaker's own video.
- It asks about information the speaker could plausibly want but could not know from their own view.
- It requires another user's visual, temporal, or contextual evidence.
- Removing any required user makes the answer unknowable, ambiguous, or much less supported.
- It is not merely two videos that happen to be near the same timestamp.
- It is not just a generic comparison of what two people did.
- It avoids template-like wording.
- The answer is grounded in visible video evidence.
- The question does not leak the answer.
- The options are plausible, mutually exclusive, and not trivially eliminated.
- The question sounds like a realistic AR assistant or memory question.
- The wording respects speaker entitlement: the speaker should not claim to have seen something only another user saw.
- If the QA depends on recognizing the same person/object across videos, the evidence should support that binding.
- It avoids hallucinating recordings, messages, screen contents, private details, or invisible intent.

## 8. Common Failure Cases

Template collapse:

- Later 32B human-review runs sometimes repeated the same shape, such as "I was focused on X; who took over / who started Y?"
- `review_10901928_32b_diversityguard_humanreview` is a clear example: most rows collapsed into role-handoff/task-coordination wording.

Two users present but no real dependency:

- The model may ask about both users, but the answer is visible from one user's video or is just a comparison.
- Example pattern from `review_10781448_failed`: "what was the other person doing nearby?" This was rejected because it did not create a real cross-view missing-detail dependency.

Self-only answerable:

- Answerability eval often rejected candidates where Alice alone or Jake alone could answer correctly.
- Example from `review_10781448_failed`: "After I stood up and walked toward the window, what did Alice do with her phone?" Alice's own video alone could answer.

Answer/evidence mismatch:

- The QA may state an answer that is not clearly supported by the cited clip, especially for object handoff or object state changes.

Weak speaker anchor:

- The question claims "I was doing X" but that event is vague, not visible, or unrelated to the missing detail.
- Example from `review_10994195_visual_truth_sameplace_qa10_humanreview`, row 8: the speaker's box/bag/courtyard anchor did not cleanly justify asking about a berry container that mainly appeared in Lucia's view.

Same-place / same-event binding problems:

- The QA assumes the same person, room, projection, or social event across two videos without enough visual continuity.
- Example from `review_10994195_visual_truth_sameplace_qa10_humanreview`, row 10: a projection-screen social reaction question needed stronger same-room/same-event binding.

LLM hallucination:

- The model can hallucinate recordings, screen content, messages, object identities, or social intent.
- Later prompt branches added privacy and visual-truth rules because of this.

Judge too wide or too strict:

- Earlier judge settings let generic questions through.
- Strict answerability can also reject potentially good but ambiguous candidates.
- Some later human-review pools intentionally skipped answerability, so those rows must not be interpreted as automatically valid.

Timestamp/user alignment:

- Current main groups by day/time token. Same timestamp does not guarantee meaningful interaction.
- Later time-stratified sampling improved coverage but did not by itself solve semantic dependency.

Prompt rule overload:

- Adding many rules improved safety and grounding but sometimes made the generated questions rigid or repetitive.
- Removing rules improved naturalness but made quality less stable.

## 9. Current Outputs / Example Files

Tracked output:

- `egolife_two_user_qa/outputs/pilot_20/README.md` is tracked as a placeholder/description.

Local ignored outputs worth checking:

- `egolife_two_user_qa/outputs/review_10781448_failed`
  Old failed prompt-zero/reviewgate run. Useful failure analysis. Zero accepted.
- `egolife_two_user_qa/outputs/review_10782267_current_main_noschema`
  Current-main-ish run with accepted examples, but not final benchmark.
- `egolife_two_user_qa/outputs/review_10782410_prompt_zero_noschema`
  Prompt-zero comparison run.
- `egolife_two_user_qa/outputs/review_10840298_diversity_accepted`
  Diversity prompt attempt.
- `egolife_two_user_qa/outputs/review_10846464_stylehard_accepted`
  Style-hard attempt. Useful, but later notes mention duplicate/template issues.
- `egolife_two_user_qa/outputs/review_10900698_32b_relaxed_humanreview`
  Early 32B relaxed human-review pool. Mostly memory_gap/task_coordination/object-state pattern.
- `egolife_two_user_qa/outputs/review_10901928_32b_diversityguard_humanreview`
  Shows role-handoff/template collapse.
- `egolife_two_user_qa/outputs/review_10902789_32b_diversitylock2_humanreview`
  Better diversity, but repeated local templates and adjacent evidence windows.
- `egolife_two_user_qa/outputs/review_10904746_32b_timestrat_pool30_humanreview`
  Larger 30-row candidate pool with time-stratified sampling. Recommended as a candidate source, not final labels.
- `egolife_two_user_qa/outputs/review_10993592_visual_truth_privacy_qa10_humanreview`
  Visual-truth/privacy prompt update. Candidate pool only.
- `egolife_two_user_qa/outputs/review_10994195_visual_truth_sameplace_qa10_humanreview`
  Later same-place prompt hardening. Validation passed, but manual quality was flagged WARN for known rows.

Recommended files inside each review bundle:

- `summary.json`
- `accepted_review.md`
- `accepted_review.html`
- `accepted_json/`
- `prompts/`
- `raw/`
- `MANUAL_REVIEW_README.md`, if present

Old/stale files:

- `egolife_two_user_qa/notebooks/two_user_egolife_qa_demo.ipynb` appears older.
- `egolife_two_user_qa/notebooks/video_first_two_user_qa_colab.ipynb` is closer to current, but check clone URL and branch before using.
- `video_generation_prompt.html` is useful for reading, but may not exactly match the latest branch prompt.

Git/GitHub state from the local scan:

- Current handoff work is based on `origin/main`.
- Before this handoff branch, local `codex/egoeverything-repro-20260620` had an upstream marked gone, but its tree matched `origin/main`.
- Local `main` was stale relative to `origin/main`.
- The unmerged branch `codex/egolife-qa-32b-relaxed-human-review` is important for latest prompt work.
- Local untracked files in the nested repo were only `.DS_Store` files.

## 10. Minimal Handoff Summary

- The real project repo is `Long-video-understanding`, not the top-level `NYU` folder.
- The current main two-user QA code is in `egolife_two_user_qa/`.
- The formal main path is video-first: manifest -> evidence -> generate/judge/answerability -> validate -> human review.
- Generator, judger, and answerability prompts are all in `egolife_two_user_qa/prompts.py`.
- The main run script is `scripts/run_qwen3vl_gpu.sh`.
- The strict final output should pass both judge and answerability gates, but human review is still required.
- Most real review outputs are local ignored files under `egolife_two_user_qa/outputs/review_*`.
- The best recent prompt ideas are on `codex/egolife-qa-32b-relaxed-human-review`, not merged to main.
- The current candidate pools are not final benchmark labels.
- Next step: choose/merge the intended prompt branch, curate human-approved examples, and freeze a reproducible final QA set.

## Suggested Clean Handoff Folder

The repo currently mixes main code, old experiments, baseline code, ignored local outputs, and unmerged prompt branches. A clean handoff folder would help:

```text
handoff/
  README.md
  pipeline_overview.md
  prompts/
    current_main_prompts.md
    relaxed_human_review_branch_notes.md
  examples/
    selected_good_examples.md
    selected_failure_cases.md
  outputs_index.md
  failure_cases.md
  next_steps.md
```

Do not copy large video/cache assets into git. Instead, track small summaries, exact evidence IDs, prompt snapshots, selected QA rows, manual review decisions, and pointers to Torch scratch locations.
