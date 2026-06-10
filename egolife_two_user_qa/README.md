# EgoLife Two-User QA Pilot

This module builds a 20-item pilot set of multiple-choice QA examples from EgoLife videos. Each question is intended to require evidence from at least two users' egocentric streams.

The default model target is `Qwen/Qwen3-VL-8B-Instruct`. No OpenRouter/Gemini API key is used. `HF_TOKEN` is optional and is only used by Hugging Face downloads.

## Local CPU Dry Run

The dry run validates Hugging Face manifest construction, evidence packet preparation, prompt creation, and schema tooling. It does not load Qwen3-VL.

```bash
python -m egolife_two_user_qa build_manifest \
  --days DAY1 \
  --agents A1_JAKE,A2_ALICE \
  --max-per-agent-day 2 \
  --output egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json

python -m egolife_two_user_qa prepare_evidence \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json \
  --output egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --target-count 1 \
  --frames-per-clip 2

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
python -m egolife_two_user_qa generate_qa \
  --backend openai-compatible-local \
  --base-url http://127.0.0.1:8000/v1 \
  --evidence egolife_two_user_qa/outputs/pilot_20/evidence_manifest.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20/qa_mcq.raw.jsonl
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
