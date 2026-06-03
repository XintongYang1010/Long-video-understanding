"""
Fixed single-agent inference for MA-EgoQA.

This ablation answers every question using only one fixed person's retrieved
30-second caption stream. It keeps the benchmark and scoring identical to the
multi-agent setting, but removes shared-memory and planner context.
"""
import argparse
import os
import time
from typing import Any

from google import genai
from google.genai import types
from tqdm import tqdm

from egomas.src.index_bm25 import BM25TextRetriever
from egomas.src.retrieval_helpers import (
    DEFAULT_AGENT_MEMORY_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    DEFAULT_SCORE_THRESHOLD,
    retrieve_for_person,
)
from egomas.utils.constants import PERSON_NAMES
from egomas.utils.io import load_benchmark, save_json
from egomas.utils.parsing import get_prediction_index
from egomas.utils.prompt_helpers import build_question_prompt

BENCHMARK_PATH = "data/MA-EgoQA.json"
OUTPUT_DIR = "results_fixed_one"
BM25_PKL_PATH = "data/30sec_bm25.pkl"
MODEL_NAME = "models/gemini-2.5-flash"

ANSWER_MAX_TOKENS = 4096
AGENT_MEMORY_TOP_K = DEFAULT_AGENT_MEMORY_TOP_K


def call_model_with_retries(client: genai.Client, prompt: str) -> str:
    """Call Gemini and retry transient overload/rate-limit failures."""
    max_retries = int(os.getenv("EGOMAS_API_RETRIES", "8"))
    retry_sleep = float(os.getenv("EGOMAS_RETRY_SLEEP", "30"))
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=ANSWER_MAX_TOKENS),
            )
            return (response.candidates[0].content.parts[0].text or "").strip()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep * (attempt + 1))
    return ""


def build_fixed_agent_prompt(agent: str, context: str, question_prompt: str) -> str:
    return f"""Answer only the header of the correct option letter (A, B, C, D, E) in the question.

You may only use {agent}'s egocentric caption context below. Do not use or infer from any other person's stream.

### {agent}'s Context
{context}

### Question
{question_prompt}

Answer only the header of the correct answer. Do not provide any other reasoning. Your answer should be a single letter (A, B, C, D, E)."""


def process_item(
    elem: dict[str, Any],
    agent: str,
    client: genai.Client,
    retriever: BM25TextRetriever,
) -> tuple[dict[str, Any], bool]:
    question_prompt = build_question_prompt(elem["question"], elem["options"])
    context = retrieve_for_person(
        retriever,
        agent,
        elem["question"],
        top_k=AGENT_MEMORY_TOP_K,
        top_k_retrieve=DEFAULT_RETRIEVE_TOP_K,
        top_k_fallback=DEFAULT_RETRIEVE_TOP_K_FALLBACK,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
    )
    prompt = build_fixed_agent_prompt(agent, context, question_prompt)
    pred = call_model_with_retries(client, prompt)

    pred_idx = get_prediction_index(pred)
    gt_idx = elem["options"].index(elem["answer"])
    correct = pred_idx == gt_idx
    return {**elem, "fixed_agent": agent, "retrieved_context": context, "pred": pred}, correct


def run_agent(
    agent: str,
    benchmark_data: list[dict],
    client: genai.Client,
    retriever: BM25TextRetriever,
    output_dir: str,
    limit: int | None = None,
) -> float:
    data = benchmark_data[:limit] if limit else benchmark_data
    results = []
    correct_count = 0

    for elem in tqdm(data, total=len(data), desc=f"Fixed {agent}"):
        elem_out, is_correct = process_item(elem, agent, client, retriever)
        results.append(elem_out)
        save_json(results, os.path.join(output_dir, f"{agent.lower()}_fixed_one.json"))
        if is_correct:
            correct_count += 1
        tqdm.write(f"{agent} accuracy: {correct_count / len(results):.2%}")

    accuracy = correct_count / len(results) if results else 0.0
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="all", choices=["all", *PERSON_NAMES])
    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH)
    parser.add_argument("--bm25-path", default=BM25_PKL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    retriever = BM25TextRetriever.load_vectorized_format(args.bm25_path)
    benchmark_data = load_benchmark(args.benchmark_path)

    agents = PERSON_NAMES if args.agent == "all" else [args.agent]
    summary = {}
    for agent in agents:
        summary[agent] = run_agent(
            agent,
            benchmark_data,
            client,
            retriever,
            args.output_dir,
            limit=args.limit,
        )

    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    print("Fixed-one summary:")
    for agent, accuracy in summary.items():
        print(f"{agent}: {accuracy:.2%}")


if __name__ == "__main__":
    main()
