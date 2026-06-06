# MA-EgoQA: Multi-Agent Egocentric Video Question Answering

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ma-egoqa.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09827-b31b1b.svg)](https://arxiv.org/abs/2603.09827)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/KangsanKim71/MA-EgoQA)

## Overview

**MA-EgoQA** is the first benchmark for question answering over multiple long-horizon egocentric video streams from embodied agents. As intelligent agents increasingly assist our physical activities, understanding events collectively observed by multiple agents becomes essential — yet remains largely unexplored.

MA-EgoQA is built on the [EgoLife](https://egolife-dataset.github.io/) dataset, where **6 people** lived together for **7 days** wearing egocentric cameras, resulting in **266 hours** of multi-agent video. Every question requires reasoning across **more than two agents' observations**.

<p align="center">
  <img src="assets/concept_figure.png" width="80%" alt="MA-EgoQA Concept Figure"/>
</p>

---

## MA-EgoQA Benchmark

### Five Question Categories

| Category | Abbr. | Description |
|---|---|---|
| Social Interaction | SI | Localizing conversations and group behaviors across video streams |
| Task Coordination | TC | How agents divide roles and collaborate toward shared goals |
| Theory of Mind | ToM | Reasoning about agents' beliefs, intentions, and mental states |
| Temporal Reasoning | TR | Concurrency and ordering of events across agents' timelines |
| Environmental Interaction | EI | Tracking distributed object usage across agents |

<p align="center">
  <img src="assets/examples.png" width="80%" alt="MA-EgoQA QA Examples"/>
</p>

---

## EgoMAS Baseline

We propose **EgoMAS** (Egocentric Multi-Agent System), a training-free baseline that addresses the unique challenges of multi-agent egocentric reasoning.

<p align="center">
  <img src="assets/egomas.png" width="80%" alt="EgoMAS Method Figure"/>
</p>

## Run EgoMAS

### 1. Clone the Repository and Install
```sh
git clone https://github.com/KangsanKim07/MA-EgoQA.git
cd MA-EgoQA
pip install -r requirements.txt
```

### 2. Download the MA-EgoQA dataset from HuggingFace to the `data/` directory
```sh
huggingface-cli download KangsanKim71/MA-EgoQA --local-dir data --repo-type dataset
```

### 3. Construct Event-based Shared Memory with 10-min Window
```sh
python egomas/src/construct_shared_memory.py
```
Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` before running this step. On shared machines, you can limit parallel API calls with `EGOMAS_NUM_WORKERS`, for example `EGOMAS_NUM_WORKERS=2`.

### 4. Index Captions with BM25
```sh
python -m egomas.src.index_bm25
```

### 5. Inference EgoMAS with Agent-wise Dynamic Retrieval
```sh
python -m egomas.src.inference_egomas  # Multi-process
```
You can run EgoMAS with a single process by run:
```sh
python -m egomas.src.inference_egomas_singleproc
```

### Fixed Single-Agent Evaluation
To evaluate a lower-bound ablation where every question is answered from only one fixed agent's 30-second caption stream:
```sh
python -m egomas.src.inference_egomas_fixed_one --agent Jake
```
Run all six fixed agents:
```sh
python -m egomas.src.inference_egomas_fixed_one --agent all
```
This keeps the multi-agent EgoMAS scripts unchanged.


### Qwen3VL Fixed-Source Ablation

This repository snapshot also includes a Qwen-only fixed-source ablation runner. It does not modify the Gemini EgoMAS scripts. It reuses the official MA-EgoQA questions and the existing 30-second caption BM25 index, so it does not require downloading full videos.

```sh
# Default Lucia/Tasha/Shure singles and pairs
QWEN3VL_LIMIT=none sbatch hpc/run_qwen3vl_subset_h200.sbatch

# Custom source combinations
QWEN3VL_LIMIT=none \
QWEN3VL_CONDITIONS="Jack Alice Katrina Jack_Alice Jack_Katrina Alice_Katrina" \
sbatch hpc/run_qwen3vl_subset_h200.sbatch
```

Summaries are produced with:

```sh
python -m egomas.src.summarize_qwen3vl_subset --output-root outputs/qwen3vl_subset
```

The 2026-06-04 full-run artifacts are stored in `../results/2026-06-04_qwen3vl_maegoqa/`.


### Qwen3VL Route B: SigLIP Frame Retrieval

Route B is a Qwen-only visual-evidence pipeline for Day 1 MA-EgoQA. It keeps the Route A experiment definition but replaces uniform frame budgets with question-aware visual retrieval.

Pipeline:

```text
QA context timeframe
-> locate clips by agent/day/time
-> dense-sample candidate frames from each selected agent
-> SigLIP encode candidate frames
-> SigLIP encode question + options
-> select per-agent top-k frames with MMR + temporal NMS
-> top-k frames + question + options into Qwen3-VL
-> A/B/C/D/E prediction, accuracy, latency
```

Important comparison detail: Route B uses per-agent top-k. Therefore `top_k=5` is comparable to Route A `frame_modes=5` as a per-agent budget:

```text
single top_k=5 -> 5 total frames
pair top_k=5   -> 10 total frames
all top_k=5    -> 5 * number_of_relevant_agents total frames
```

Core files:

```text
egomas/src/evaluate_day1_qwen3vl_routeb_retrieval.py
egomas/src/merge_routeb_retrieval_results.py
hpc/run_routeb_siglip_h200.sbatch
```

Current status: a one-question H200 smoke run has completed successfully with `top_k=5`, `single/pair/all`, and chunk-local retrieval plus Qwen inference.

Smoke test on one H200 GPU:

```sh
cd /scratch/$USER/github_sync_long_video_understanding/MA-EgoQA
mkdir -p /scratch/$USER/data/multiresult/routeB_siglip_day1_top5_chunklocal

ROUTEB_LIMIT=1 \
ROUTEB_TOP_KS="5" \
ROUTEB_RETRIEVAL_BATCH_SIZE=256 \
ROUTEB_OUTPUT_PATH=/scratch/$USER/data/multiresult/routeB_siglip_day1_top5_chunklocal/smoke_top5_chunklocal.json \
ROUTEB_SAVE_EVERY=1 \
sbatch --time=01:55:00 hpc/run_routeb_siglip_h200.sbatch
```

Formal Day 1 top-k=5 runs should be submitted in small chunks so each job retrieves and answers locally instead of building a global frame index:

```sh
ROUTEB_START_INDEX=0 \
ROUTEB_LIMIT=10 \
ROUTEB_TOP_KS="5" \
ROUTEB_RETRIEVAL_BATCH_SIZE=256 \
ROUTEB_OUTPUT_PATH=/scratch/$USER/data/multiresult/routeB_siglip_day1_top5_chunklocal/chunk_0_9.json \
ROUTEB_SAVE_EVERY=10 \
sbatch --time=01:55:00 hpc/run_routeb_siglip_h200.sbatch
```

After all chunks complete:

```sh
python -m egomas.src.merge_routeb_retrieval_results \
  --input-glob "/scratch/$USER/data/multiresult/routeB_siglip_day1_top5_chunklocal/chunk_*.json" \
  --output-path /scratch/$USER/data/multiresult/routeB_siglip_day1_top5_chunklocal/merged_top5.json \
  --strict
```

---

## Citation

```bibtex
@misc{kim2026maegoqa,
    title={MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents}, 
    author={Kangsan Kim and Yanlai Yang and Suji Kim and Woongyeong Yeo and Youngwan Lee and Mengye Ren and Sung Ju Hwang},
    year={2026},
    eprint={2603.09827},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.09827}, 
}
```
