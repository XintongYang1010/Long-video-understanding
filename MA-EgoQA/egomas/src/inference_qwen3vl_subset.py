"""Qwen3-VL subset inference for MA-EgoQA fixed-agent ablations.

This runner intentionally does not import or modify the Gemini inference files.
It mirrors the fixed single-agent setup: retrieve 30-second captions for a fixed
agent set, ask a multiple-choice model for one option letter, and score against
the official MA-EgoQA answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from egomas.utils.prompt_helpers import build_question_prompt
from egomas.utils.retrieval_format import format_retrieved_context


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PROJECT_ROOT = Path(
    os.getenv("QWEN3VL_PROJECT_ROOT", Path(__file__).resolve().parents[3])
)
DEFAULT_DATA_ROOT = Path(
    os.getenv(
        "QWEN3VL_MAEGOQA_DATA_ROOT",
        DEFAULT_PROJECT_ROOT / "ma_egoqa_reproduce" / "MA-EgoQA" / "data",
    )
)
DEFAULT_BENCHMARK_PATH = DEFAULT_DATA_ROOT / "MA-EgoQA.json"
DEFAULT_BM25_PATH = DEFAULT_DATA_ROOT / "30sec_bm25.pkl"
DEFAULT_OUTPUT_ROOT = Path("outputs/qwen3vl_subset")

AGENT_ALIASES = {
    "Jack": "Jake",
    "Jake": "Jake",
    "Alice": "Alice",
    "Tasha": "Tasha",
    "Lucia": "Lucia",
    "Katrina": "Katrina",
    "Shure": "Shure",
}
VALID_OPTIONS = ("a", "b", "c", "d", "e")
OPTION_LETTERS = "ABCDE"

DEFAULT_AGENT_MEMORY_TOP_K = 5
DEFAULT_RETRIEVE_TOP_K = 100
DEFAULT_RETRIEVE_TOP_K_FALLBACK = 1000
DEFAULT_SCORE_THRESHOLD = 10.0


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_model(model_id: str):
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
        log("model_class=Qwen3VLForConditionalGeneration")
    except Exception:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
        log("model_class=AutoModelForImageTextToText")

    kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    try:
        return model_cls.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)


class QwenTextGenerator:
    def __init__(self, model_id: str, max_new_tokens: int):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        start = time.time()
        log(f"loading_processor={model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        log(f"loading_model={model_id}")
        self.model = load_model(model_id)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        log(f"model_first_param_device={self.device}")
        log(f"model_loaded_seconds={time.time() - start:.1f}")

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated, strict=False)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class PickledBM25Retriever:
    """Load the existing 30sec_bm25.pkl without importing pandas-heavy helpers."""

    def __init__(self, file_paths: list[str], captions: list[str], bm25: Any):
        self.file_paths = file_paths
        self.captions = captions
        self.bm25 = bm25

    @classmethod
    def load(cls, path: str | Path) -> "PickledBM25Retriever":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            file_paths=data["file_paths"],
            captions=data["captions"],
            bm25=data["bm25"],
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split(" ")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False,
    ):
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        ids = [self.file_paths[idx] for idx in top_indices]
        captions = [self.captions[idx] for idx in top_indices]
        if return_scores:
            return ids, captions, [float(scores[idx]) for idx in top_indices]
        return ids, captions


def normalize_agent_name(name: str) -> str:
    key = name.strip()
    if key in AGENT_ALIASES:
        return AGENT_ALIASES[key]
    titled = key.title()
    if titled in AGENT_ALIASES:
        return AGENT_ALIASES[titled]
    allowed = ", ".join(sorted(AGENT_ALIASES))
    raise ValueError(f"Unsupported agent {name!r}. Allowed agents: {allowed}")


def parse_condition(condition: str) -> list[str]:
    agents = [
        normalize_agent_name(part)
        for part in re.split(r"[_+,]", condition)
        if part.strip()
    ]
    if not agents:
        raise ValueError("Condition must contain at least one agent name")
    if len(set(agents)) != len(agents):
        raise ValueError(f"Condition contains duplicate agents: {condition!r}")
    return agents


def retrieve_for_person(
    retriever: PickledBM25Retriever,
    name: str,
    query: str,
    top_k: int = DEFAULT_AGENT_MEMORY_TOP_K,
    top_k_retrieve: int = DEFAULT_RETRIEVE_TOP_K,
    top_k_fallback: int = DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    name_upper = name.upper()
    results, captions, scores = retriever.retrieve(
        query,
        top_k=top_k_retrieve,
        return_scores=True,
    )
    named_results = [
        {"id": rid, "caption": cap, "score": sc}
        for rid, cap, sc in zip(results, captions, scores, strict=False)
        if name_upper in rid
    ]
    if len(named_results) < top_k:
        results, captions, scores = retriever.retrieve(
            query,
            top_k=top_k_fallback,
            return_scores=True,
        )
        named_results = [
            {"id": rid, "caption": cap, "score": sc}
            for rid, cap, sc in zip(results, captions, scores, strict=False)
            if name_upper in rid
        ]
    return format_retrieved_context(
        named_results,
        top_k=top_k,
        score_threshold=score_threshold,
    )


def build_subset_prompt(
    agents: list[str],
    retrieved_contexts: dict[str, str],
    question_prompt: str,
) -> str:
    agent_text = " and ".join(agents)
    sections = []
    for agent in agents:
        context = retrieved_contexts.get(agent, "").strip() or "No retrieved context."
        sections.append(f"### {agent}'s Context\n{context}")
    context_block = "\n\n".join(sections)
    return f"""Answer only the header of the correct option letter (A, B, C, D, E) in the question.

You may only use the egocentric caption context from {agent_text} below. Do not use or infer from any other person's stream. If the context is incomplete, choose the best-supported option from the provided choices.

{context_block}

### Question
{question_prompt}

Answer only the header of the correct answer. Do not provide any other reasoning. Your answer should be a single letter (A, B, C, D, E)."""


def normalize_prediction(pred: str) -> str:
    if not pred:
        return ""
    cleaned = pred.strip().lower()
    cleaned = cleaned.replace("the correct answer is", "")
    cleaned = cleaned.replace("answer:", "")
    cleaned = cleaned.replace("option", "")
    match = re.search(r"\b([abcde])\b", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"[abcde]", cleaned)
    return match.group(0) if match else ""


def prediction_index(pred: str) -> int:
    normalized = normalize_prediction(pred)
    if normalized in VALID_OPTIONS:
        return VALID_OPTIONS.index(normalized)
    return -1


def ground_truth_index(elem: dict[str, Any]) -> int:
    answer = str(elem.get("answer", "")).strip()
    options = elem.get("options", [])
    if answer in options:
        return options.index(answer)
    normalized = normalize_prediction(answer)
    if normalized in VALID_OPTIONS:
        return VALID_OPTIONS.index(normalized)
    raise ValueError(f"Could not map answer to an option: {answer!r}")


@dataclass
class RunStats:
    condition: str
    agents: list[str]
    num_items: int = 0
    correct: int = 0
    invalid_predictions: int = 0
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.num_items if self.num_items else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "agents": self.agents,
            "num_items": self.num_items,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "invalid_predictions": self.invalid_predictions,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_condition(args: argparse.Namespace) -> None:
    agents = parse_condition(args.condition)
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_ROOT / args.condition)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if predictions_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{predictions_path} already exists. Use --overwrite or choose another output dir."
        )
    if predictions_path.exists():
        predictions_path.unlink()

    log(f"hostname={socket.gethostname()}")
    log(f"python={sys.executable}")
    log(f"condition={args.condition}")
    log(f"agents={','.join(agents)}")
    log(f"cuda_available={torch.cuda.is_available()}")
    log(f"cuda_device_count={torch.cuda.device_count()}")
    log(f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this inside a GPU Slurm job.")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        log(f"gpu[{idx}]={props.name}; total_memory_gb={props.total_memory / 1024**3:.1f}")

    benchmark_data = load_json(args.benchmark_path)
    if args.limit:
        benchmark_data = benchmark_data[: args.limit]
    log(f"num_items={len(benchmark_data)}")

    retriever = PickledBM25Retriever.load(args.bm25_path)
    generator = QwenTextGenerator(args.model_id, args.max_new_tokens)

    stats = RunStats(condition=args.condition, agents=agents)
    start = time.time()
    for item_index, elem in enumerate(
        tqdm(benchmark_data, total=len(benchmark_data), desc=args.condition)
    ):
        question_prompt = build_question_prompt(elem["question"], elem["options"])
        retrieved_contexts = {
            agent: retrieve_for_person(
                retriever,
                agent,
                elem["question"],
                top_k=args.agent_memory_top_k,
                top_k_retrieve=args.retrieve_top_k,
                top_k_fallback=args.retrieve_top_k_fallback,
                score_threshold=args.score_threshold,
            )
            for agent in agents
        }
        prompt = build_subset_prompt(agents, retrieved_contexts, question_prompt)
        pred = generator.generate(prompt)
        pred_idx = prediction_index(pred)
        gt_idx = ground_truth_index(elem)
        correct = pred_idx == gt_idx

        stats.num_items += 1
        stats.correct += int(correct)
        stats.invalid_predictions += int(pred_idx < 0)
        stats.elapsed_seconds = time.time() - start

        row = {
            "item_index": item_index,
            "condition": args.condition,
            "agents": agents,
            "question": elem["question"],
            "options": elem["options"],
            "answer": elem["answer"],
            "gt_idx": gt_idx,
            "gt_letter": OPTION_LETTERS[gt_idx] if 0 <= gt_idx < len(OPTION_LETTERS) else "",
            "pred": pred,
            "pred_idx": pred_idx,
            "pred_letter": OPTION_LETTERS[pred_idx] if 0 <= pred_idx < len(OPTION_LETTERS) else "",
            "correct": correct,
            "retrieved_contexts": retrieved_contexts,
        }
        append_jsonl(predictions_path, row)
        write_json(summary_path, stats.to_dict())
        log(
            f"{args.condition} item={stats.num_items} "
            f"accuracy={stats.accuracy:.2%} pred={row['pred_letter'] or 'INVALID'} "
            f"gt={row['gt_letter']}"
        )

    stats.elapsed_seconds = time.time() - start
    write_json(summary_path, stats.to_dict())
    log(f"saved_predictions={predictions_path}")
    log(f"saved_summary={summary_path}")
    log(f"final_accuracy={stats.accuracy:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        help="Single agent or agent combination, e.g. Lucia or Jack_Alice_Katrina.",
    )
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--agent-memory-top-k", type=int, default=DEFAULT_AGENT_MEMORY_TOP_K)
    parser.add_argument("--retrieve-top-k", type=int, default=DEFAULT_RETRIEVE_TOP_K)
    parser.add_argument(
        "--retrieve-top-k-fallback",
        type=int,
        default=DEFAULT_RETRIEVE_TOP_K_FALLBACK,
    )
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_condition(parse_args())


if __name__ == "__main__":
    main()
