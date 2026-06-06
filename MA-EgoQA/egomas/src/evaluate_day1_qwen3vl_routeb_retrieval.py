"""
Evaluate DAY1 MA-EgoQA with Route B visual retrieval.

Route B keeps Route A's experiment definition:
  * DAY1 structured context windows
  * relevant agents listed in each context
  * single / pair / all groups generated from those relevant agents

The only change is frame selection. Instead of uniform per-agent sampling,
Route B densely samples candidate frames, ranks them with a SigLIP text-image
retriever, keeps per-agent top-k frames, and sends those frames to Qwen3-VL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from egomas.src.evaluate_day1_qwen3vl_frames import (
    AGENT_TO_FOLDER,
    BENCHMARK_PATH,
    MODEL_NAME,
    VIDEO_ROOT,
    ContextWindow,
    FrameSpec,
    VideoSegment,
    build_question_prompt,
    build_video_index,
    is_day1_item,
    load_frame_dependencies,
    load_qwen_model,
    overlapping_segments,
    parse_contexts,
    relevant_agents,
    run_qwen_answer,
    trial_groups,
)
from egomas.utils.io import load_json, save_json
from egomas.utils.parsing import get_prediction_index


OUTPUT_PATH = "/scratch/xy3257/data/multiresult/routeB_siglip_day1.json"
RETRIEVER_MODEL = "google/siglip-base-patch16-224"
AGENT_ORDER = ["JAKE", "ALICE", "KATRINA", "LUCIA", "TASHA", "SHURE"]
AGENT_RANK = {agent: idx for idx, agent in enumerate(AGENT_ORDER)}


@dataclass(frozen=True)
class CandidateFrame:
    spec: FrameSpec
    candidate_rank: int


@dataclass(frozen=True)
class RankedFrame:
    candidate: CandidateFrame
    score: float
    rank: int


def option_query_text(item: dict[str, Any]) -> str:
    options = "\n".join(
        f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(item["options"])
    )
    return f"Question: {item['question']}\nOptions:\n{options}"


def iter_dense_offsets(local_start: float, local_end: float, candidate_fps: float) -> list[float]:
    duration = max(local_end - local_start, 0.0)
    if duration <= 0:
        return []
    step = 1.0 / candidate_fps
    offsets: list[float] = []
    pos = local_start + step / 2.0
    while pos < local_end:
        offsets.append(pos)
        pos += step
    if not offsets:
        offsets.append(local_start + duration / 2.0)
    return offsets


def build_agent_candidate_specs(
    video_index: dict[str, list[VideoSegment]],
    windows: list[ContextWindow],
    agent: str,
    candidate_fps: float,
    max_candidates: int,
) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    seen: set[tuple[str, str, int]] = set()
    for window in windows:
        if agent not in window.agents:
            continue
        for segment in overlapping_segments(video_index, agent, window):
            local_start = max(window.start_sec - segment.start_sec, 0.0)
            local_end = min(window.end_sec - segment.start_sec, segment.end_sec - segment.start_sec)
            for offset in iter_dense_offsets(local_start, local_end, candidate_fps):
                timestamp_sec = segment.start_sec + offset
                dedup_key = (window.key, str(segment.path), int(round(offset * 1000)))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                candidates.append(
                    CandidateFrame(
                        spec=FrameSpec(
                            agent=agent,
                            context_key=window.key,
                            video_path=str(segment.path),
                            video_start_token=segment.start_token,
                            offset_sec=round(offset, 3),
                            timestamp_sec=round(timestamp_sec, 3),
                        ),
                        candidate_rank=len(candidates),
                    )
                )
    candidates.sort(
        key=lambda item: (
            item.spec.timestamp_sec,
            item.spec.agent,
            item.spec.video_path,
            item.spec.offset_sec,
        )
    )
    if max_candidates > 0 and len(candidates) > max_candidates:
        if max_candidates == 1:
            return [candidates[len(candidates) // 2]]
        stride = (len(candidates) - 1) / (max_candidates - 1)
        keep_indices = sorted({int(round(idx * stride)) for idx in range(max_candidates)})
        candidates = [candidates[idx] for idx in keep_indices]
    return [
        CandidateFrame(spec=candidate.spec, candidate_rank=idx)
        for idx, candidate in enumerate(candidates)
    ]


def decode_image_for_spec(spec: FrameSpec) -> Any | None:
    cv2, Image = load_frame_dependencies()
    cap = cv2.VideoCapture(spec.video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = max(int(round(spec.offset_sec * fps)), 0)
        if frame_count > 0:
            frame_idx = min(frame_idx, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def load_images_for_specs(specs: list[FrameSpec]) -> tuple[list[Any], list[FrameSpec]]:
    images: list[Any] = []
    kept_specs: list[FrameSpec] = []
    for spec in specs:
        image = decode_image_for_spec(spec)
        if image is None:
            continue
        images.append(image)
        kept_specs.append(spec)
    return images, kept_specs


def decode_images_for_candidates(
    candidates: list[CandidateFrame],
) -> tuple[list[Any], list[CandidateFrame]]:
    cv2, Image = load_frame_dependencies()
    grouped: dict[str, list[tuple[int, CandidateFrame]]] = {}
    for idx, candidate in enumerate(candidates):
        grouped.setdefault(candidate.spec.video_path, []).append((idx, candidate))

    decoded: list[tuple[int, Any, CandidateFrame]] = []
    for video_path, entries in grouped.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for original_idx, candidate in sorted(entries, key=lambda item: item[1].spec.offset_sec):
                frame_idx = max(int(round(candidate.spec.offset_sec * fps)), 0)
                if frame_count > 0:
                    frame_idx = min(frame_idx, frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded.append((original_idx, Image.fromarray(rgb), candidate))
        finally:
            cap.release()

    decoded.sort(key=lambda item: item[0])
    return [item[1] for item in decoded], [item[2] for item in decoded]


def choose_temporal_candidates(
    candidates: list[CandidateFrame],
    top_k: int,
) -> list[RankedFrame]:
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return [
            RankedFrame(candidate=candidate, score=0.0, rank=rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]
    stride = (len(candidates) - 1) / (top_k - 1) if top_k > 1 else 0.0
    indices = sorted({int(round(idx * stride)) for idx in range(top_k)})
    selected = [candidates[idx] for idx in indices]
    return [
        RankedFrame(candidate=candidate, score=0.0, rank=rank)
        for rank, candidate in enumerate(selected, start=1)
    ]


class SiglipRetriever:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _to_device(self, inputs: Any) -> Any:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _feature_tensor(self, value: Any, kind: str) -> Any:
        if hasattr(value, "float"):
            return value
        for attr in (f"{kind}_embeds", "pooler_output"):
            feature = getattr(value, attr, None)
            if feature is not None:
                return feature
        last_hidden = getattr(value, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden.mean(dim=1)
        raise RuntimeError(f"Retriever model did not return {kind} embeddings")

    def encode_text(self, query: str) -> Any:
        inputs = self.processor(
            text=[query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)
        with self.torch.inference_mode():
            if hasattr(self.model, "get_text_features"):
                features = self.model.get_text_features(**inputs)
            else:
                features = self.model(**inputs)
            features = self._feature_tensor(features, "text")
        return self.torch.nn.functional.normalize(features.float(), dim=-1)[0].detach().cpu()

    def encode_images(self, candidates: list[CandidateFrame]) -> tuple[Any, list[CandidateFrame]]:
        embeddings: list[Any] = []
        kept_candidates: list[CandidateFrame] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            images, valid_candidates = decode_images_for_candidates(batch)
            if not images:
                continue
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = self._to_device(inputs)
            with self.torch.inference_mode():
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(**inputs)
                else:
                    features = self.model(**inputs)
                features = self._feature_tensor(features, "image")
            features = self.torch.nn.functional.normalize(features.float(), dim=-1).detach().cpu()
            embeddings.append(features)
            kept_candidates.extend(valid_candidates)
        if not embeddings:
            return self.torch.empty((0, 1)), []
        return self.torch.cat(embeddings, dim=0), kept_candidates


def violates_temporal_nms(
    candidate: CandidateFrame,
    selected: list[int],
    candidates: list[CandidateFrame],
    temporal_nms_sec: float,
) -> bool:
    if temporal_nms_sec <= 0:
        return False
    for selected_idx in selected:
        selected_spec = candidates[selected_idx].spec
        if selected_spec.video_path != candidate.spec.video_path:
            continue
        if abs(selected_spec.timestamp_sec - candidate.spec.timestamp_sec) < temporal_nms_sec:
            return True
    return False


def mmr_select(
    candidates: list[CandidateFrame],
    image_embeddings: Any,
    query_embedding: Any,
    top_k: int,
    mmr_lambda: float,
    temporal_nms_sec: float,
) -> list[RankedFrame]:
    torch = __import__("torch")
    if top_k <= 0 or not candidates:
        return []
    if image_embeddings.shape[0] != len(candidates):
        raise ValueError("Embedding count does not match candidate count")

    scores = image_embeddings @ query_embedding
    shortlist_size = min(len(candidates), max(top_k * 20, 200))
    shortlist = torch.topk(scores, k=shortlist_size).indices.tolist()
    selected: list[int] = []
    available = set(shortlist)

    while available and len(selected) < top_k:
        best_idx: int | None = None
        best_value = -math.inf
        for idx in list(available):
            candidate = candidates[idx]
            if violates_temporal_nms(candidate, selected, candidates, temporal_nms_sec):
                continue
            relevance = float(scores[idx])
            if selected:
                selected_embeddings = image_embeddings[selected]
                diversity_penalty = float((selected_embeddings @ image_embeddings[idx]).max())
            else:
                diversity_penalty = 0.0
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * diversity_penalty
            if value > best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        available.remove(best_idx)

    if len(selected) < top_k:
        for idx in torch.argsort(scores, descending=True).tolist():
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= top_k:
                break

    ranked = [
        RankedFrame(
            candidate=candidates[idx],
            score=float(scores[idx]),
            rank=rank,
        )
        for rank, idx in enumerate(selected[:top_k], start=1)
    ]
    return ranked


def rank_agent_candidates(
    candidates: list[CandidateFrame],
    query_text: str,
    retriever: SiglipRetriever | None,
    top_k_max: int,
    mmr_lambda: float,
    temporal_nms_sec: float,
) -> tuple[list[RankedFrame], float]:
    started = time.perf_counter()
    if retriever is None:
        ranked = choose_temporal_candidates(candidates, top_k_max)
        return ranked, time.perf_counter() - started
    if not candidates:
        return [], time.perf_counter() - started
    query_embedding = retriever.encode_text(query_text)
    image_embeddings, kept_candidates = retriever.encode_images(candidates)
    ranked = mmr_select(
        candidates=kept_candidates,
        image_embeddings=image_embeddings,
        query_embedding=query_embedding,
        top_k=top_k_max,
        mmr_lambda=mmr_lambda,
        temporal_nms_sec=temporal_nms_sec,
    )
    return ranked, time.perf_counter() - started


def frame_dict(ranked: RankedFrame) -> dict[str, Any]:
    data = ranked.candidate.spec.__dict__.copy()
    data["retrieval_rank"] = ranked.rank
    data["retrieval_score"] = ranked.score
    data["candidate_rank"] = ranked.candidate.candidate_rank
    return data


def canonical_agent_group(agents: list[str]) -> tuple[str, ...]:
    return tuple(sorted(agents, key=lambda agent: (AGENT_RANK.get(agent, 999), agent)))


def agent_group_label(agents: list[str] | tuple[str, ...]) -> str:
    return "_".join(canonical_agent_group(list(agents)))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": {},
        "by_top_k": {},
        "by_eval_mode": {},
        "by_top_k_eval_mode": {},
    }

    def acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answered = [row for row in rows if row.get("pred_index", -1) >= 0]
        correct = sum(1 for row in rows if row.get("correct"))
        trials = len(rows)
        return {
            "trials": trials,
            "answered": len(answered),
            "correct": correct,
            "accuracy": correct / trials if trials else 0.0,
            "avg_candidate_frames": sum(row.get("candidate_frame_count", 0) for row in rows) / trials
            if trials
            else 0.0,
            "avg_selected_frames": sum(row.get("num_frames", 0) for row in rows) / trials
            if trials
            else 0.0,
            "avg_retrieval_sec": sum(row.get("retrieval_sec", 0.0) for row in rows) / trials
            if trials
            else 0.0,
            "avg_qwen_sec": sum(row.get("qwen_sec", 0.0) for row in rows) / trials
            if trials
            else 0.0,
            "avg_total_sec": sum(row.get("elapsed_sec", 0.0) for row in rows) / trials
            if trials
            else 0.0,
            "no_frame_count": sum(1 for row in rows if row.get("num_frames", 0) == 0),
        }

    summary["overall"] = acc(results)
    top_ks = sorted({row["top_k"] for row in results})
    eval_modes = sorted({row["eval_mode"] for row in results})
    for top_k in top_ks:
        rows = [row for row in results if row["top_k"] == top_k]
        summary["by_top_k"][str(top_k)] = acc(rows)
    for eval_mode in eval_modes:
        rows = [row for row in results if row["eval_mode"] == eval_mode]
        summary["by_eval_mode"][eval_mode] = acc(rows)
    for top_k in top_ks:
        summary["by_top_k_eval_mode"][str(top_k)] = {}
        for eval_mode in eval_modes:
            rows = [
                row
                for row in results
                if row["top_k"] == top_k and row["eval_mode"] == eval_mode
            ]
            summary["by_top_k_eval_mode"][str(top_k)][eval_mode] = acc(rows)

    single_rows = [
        row
        for row in results
        if row.get("eval_mode") == "single" and len(row.get("agents", [])) == 1
    ]
    if single_rows:
        summary["by_single_agent"] = {}
        for top_k in top_ks:
            summary["by_single_agent"][str(top_k)] = {}
            agents = sorted({row["agents"][0] for row in single_rows}, key=lambda agent: (AGENT_RANK.get(agent, 999), agent))
            for agent in agents:
                rows = [
                    row
                    for row in single_rows
                    if row["top_k"] == top_k and row["agents"][0] == agent
                ]
                summary["by_single_agent"][str(top_k)][agent] = acc(rows)

    pair_rows_for_groups = [
        row
        for row in results
        if row.get("eval_mode") == "pair" and len(row.get("agents", [])) == 2
    ]
    if pair_rows_for_groups:
        summary["by_pair_agent"] = {}
        for top_k in top_ks:
            summary["by_pair_agent"][str(top_k)] = {}
            pairs = sorted(
                {canonical_agent_group(row["agents"]) for row in pair_rows_for_groups},
                key=lambda pair: tuple(AGENT_RANK.get(agent, 999) for agent in pair),
            )
            for pair in pairs:
                rows = [
                    row
                    for row in pair_rows_for_groups
                    if row["top_k"] == top_k and canonical_agent_group(row["agents"]) == pair
                ]
                summary["by_pair_agent"][str(top_k)][agent_group_label(pair)] = acc(rows)

    def oracle_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_question: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_question.setdefault(int(row["question_index"]), []).append(row)
        trials = len(by_question)
        answered = sum(
            1
            for question_rows in by_question.values()
            if any(row.get("pred_index", -1) >= 0 for row in question_rows)
        )
        correct = sum(
            1
            for question_rows in by_question.values()
            if any(row.get("correct") for row in question_rows)
        )
        return {
            "trials": trials,
            "answered": answered,
            "correct": correct,
            "accuracy": correct / trials if trials else 0.0,
            "note": "oracle upper bound across trials for the same question; latency is not a real runnable-system latency",
        }

    single_rows = [row for row in results if row.get("eval_mode") == "single"]
    pair_rows = [row for row in results if row.get("eval_mode") == "pair"]
    if single_rows or pair_rows:
        summary["oracle_best_by_top_k"] = {}
        for top_k in top_ks:
            summary["oracle_best_by_top_k"][str(top_k)] = {}
            if single_rows:
                rows = [row for row in single_rows if row["top_k"] == top_k]
                summary["oracle_best_by_top_k"][str(top_k)]["single_best"] = oracle_best(rows)
            if pair_rows:
                rows = [row for row in pair_rows if row["top_k"] == top_k]
                summary["oracle_best_by_top_k"][str(top_k)]["pair_best"] = oracle_best(rows)

    all_rows = [row for row in results if row.get("eval_mode") == "all"]
    if all_rows:
        summary["all_agents"] = {
            str(top_k): acc([row for row in all_rows if row["top_k"] == top_k])
            for top_k in top_ks
        }
    return summary


def write_summary_csv(summary: dict[str, Any], path: str | Path) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for top_k, eval_data in summary.get("by_top_k_eval_mode", {}).items():
        for eval_mode, metrics in eval_data.items():
            row = {"top_k": top_k, "group": eval_mode, "agent": ""}
            row.update(metrics)
            rows.append(row)
    for top_k, agent_data in summary.get("by_single_agent", {}).items():
        for agent, metrics in agent_data.items():
            row = {"top_k": top_k, "group": "single_agent", "agent": agent}
            row.update(metrics)
            rows.append(row)
    for top_k, pair_data in summary.get("by_pair_agent", {}).items():
        for pair, metrics in pair_data.items():
            row = {"top_k": top_k, "group": "pair_agent", "agent": pair}
            row.update(metrics)
            rows.append(row)
    for top_k, metrics in summary.get("all_agents", {}).items():
        row = {"top_k": top_k, "group": "all_agents", "agent": "ALL_CONTEXT_AGENTS"}
        row.update(metrics)
        rows.append(row)
    for top_k, best_data in summary.get("oracle_best_by_top_k", {}).items():
        for group, metrics in best_data.items():
            row = {"top_k": top_k, "group": group, "agent": ""}
            row.update(metrics)
            rows.append(row)
    fieldnames = [
        "top_k",
        "group",
        "agent",
        "trials",
        "answered",
        "correct",
        "accuracy",
        "avg_candidate_frames",
        "avg_selected_frames",
        "avg_retrieval_sec",
        "avg_qwen_sec",
        "avg_total_sec",
        "no_frame_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_outputs(payload: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, str(path))
    write_summary_csv(payload.get("summary", {}), path.with_suffix(".summary.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH)
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--retriever-model", default=RETRIEVER_MODEL)
    parser.add_argument("--eval-modes", nargs="+", default=["single", "pair", "all"], choices=["single", "pair", "all"])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--candidate-fps", type=float, default=1.0)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-agent", type=int, default=0)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--temporal-nms-sec", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of DAY1 questions; 0 means all.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset within filtered DAY1 questions.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--retriever-device", default="auto")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Build candidates and placeholder selections without loading SigLIP/Qwen.")
    args = parser.parse_args()

    if args.candidate_fps <= 0:
        raise SystemExit("--candidate-fps must be positive")
    top_ks = sorted({top_k for top_k in args.top_ks if top_k > 0})
    if not top_ks:
        raise SystemExit("--top-ks must contain at least one positive integer")
    top_k_max = max(top_ks)
    mmr_lambda = min(max(args.mmr_lambda, 0.0), 1.0)

    benchmark = load_json(args.benchmark_path)
    day1_items = [item for item in benchmark if is_day1_item(item)]
    if args.start_index:
        day1_items = day1_items[args.start_index :]
    if args.limit > 0:
        day1_items = day1_items[: args.limit]

    print(f"Loaded {len(benchmark)} benchmark items; evaluating {len(day1_items)} DAY1 items.")
    print(f"Route B top_ks: {top_ks}; eval modes: {args.eval_modes}; candidate_fps={args.candidate_fps}")
    print("Condition generation matches Route A: relevant context agents -> single/pair/all.")

    video_index = build_video_index(args.video_root, day="DAY1")
    for agent, segments in video_index.items():
        print(f"{agent}: {len(segments)} DAY1 video segments")

    retriever: SiglipRetriever | None = None
    model = processor = torch = None
    if not args.dry_run:
        import torch as torch_module

        retriever_device = args.retriever_device
        if retriever_device == "auto":
            retriever_device = "cuda" if torch_module.cuda.is_available() else "cpu"
        retriever = SiglipRetriever(
            model_name=args.retriever_model,
            device=retriever_device,
            batch_size=args.retrieval_batch_size,
        )
        model, processor, torch = load_qwen_model(
            args.model_name,
            args.device_map,
            args.min_pixels,
            args.max_pixels,
        )

    total_trials = 0
    for item in day1_items:
        agents = relevant_agents(parse_contexts(item))
        total_trials += len(top_ks) * len(trial_groups(agents, args.eval_modes))

    results: list[dict[str, Any]] = []
    running_correct = 0
    trial_count = 0
    progress = tqdm(total=total_trials, desc="DAY1 Route B Qwen3-VL eval")
    try:
        for q_idx, item in enumerate(day1_items):
            windows = parse_contexts(item)
            agents = relevant_agents(windows)
            groups = trial_groups(agents, args.eval_modes)
            gt_idx = item["options"].index(item["answer"])
            query_text = option_query_text(item)

            agent_rankings: dict[str, list[RankedFrame]] = {}
            agent_candidate_counts: dict[str, int] = {}
            agent_retrieval_sec: dict[str, float] = {}
            for agent in agents:
                candidates = build_agent_candidate_specs(
                    video_index=video_index,
                    windows=windows,
                    agent=agent,
                    candidate_fps=args.candidate_fps,
                    max_candidates=args.max_candidates_per_agent,
                )
                ranked, retrieval_sec = rank_agent_candidates(
                    candidates=candidates,
                    query_text=query_text,
                    retriever=retriever,
                    top_k_max=top_k_max,
                    mmr_lambda=mmr_lambda,
                    temporal_nms_sec=args.temporal_nms_sec,
                )
                agent_rankings[agent] = ranked
                agent_candidate_counts[agent] = len(candidates)
                agent_retrieval_sec[agent] = retrieval_sec

            for top_k in top_ks:
                for eval_mode, group_agents in groups:
                    started = time.perf_counter()
                    selected_ranked: list[RankedFrame] = []
                    for agent in group_agents:
                        selected_ranked.extend(agent_rankings.get(agent, [])[:top_k])
                    selected_ranked.sort(
                        key=lambda item: (
                            item.candidate.spec.timestamp_sec,
                            item.candidate.spec.agent,
                            item.candidate.spec.video_path,
                            item.candidate.spec.offset_sec,
                        )
                    )
                    selected_specs = [ranked.candidate.spec for ranked in selected_ranked]
                    retrieval_sec = sum(agent_retrieval_sec.get(agent, 0.0) for agent in group_agents)
                    candidate_frame_count = sum(agent_candidate_counts.get(agent, 0) for agent in group_agents)

                    qwen_started = time.perf_counter()
                    if args.dry_run:
                        images = []
                        kept_specs = selected_specs
                        raw_pred = "DRY_RUN"
                        pred_idx = -1
                    else:
                        images, kept_specs = load_images_for_specs(selected_specs)
                        if not images:
                            raw_pred = "NO_FRAMES"
                            pred_idx = -1
                        else:
                            prompt = build_question_prompt(item, group_agents, kept_specs)
                            raw_pred = run_qwen_answer(
                                model=model,
                                processor=processor,
                                torch=torch,
                                prompt=prompt,
                                images=images,
                                max_new_tokens=args.max_new_tokens,
                            )
                            pred_idx = get_prediction_index(raw_pred)
                    qwen_sec = time.perf_counter() - qwen_started
                    elapsed = time.perf_counter() - started + retrieval_sec
                    row = {
                        "question_index": args.start_index + q_idx,
                        "category": item.get("category"),
                        "subcategory": item.get("subcategory"),
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"],
                        "gt_index": gt_idx,
                        "contexts": item.get("contexts", {}),
                        "route": "B_siglip_retrieval",
                        "retriever_model": args.retriever_model if not args.dry_run else "DRY_RUN_TEMPORAL",
                        "candidate_fps": args.candidate_fps,
                        "top_k": top_k,
                        "top_k_semantics": "per_agent",
                        "eval_mode": eval_mode,
                        "agents": list(group_agents),
                        "candidate_frame_count": candidate_frame_count,
                        "candidate_frame_count_by_agent": {
                            agent: agent_candidate_counts.get(agent, 0) for agent in group_agents
                        },
                        "num_frames": len(kept_specs),
                        "frames": [frame_dict(ranked) for ranked in selected_ranked],
                        "raw_pred": raw_pred,
                        "pred_index": pred_idx,
                        "correct": pred_idx == gt_idx,
                        "retrieval_sec": retrieval_sec,
                        "qwen_sec": qwen_sec,
                        "elapsed_sec": elapsed,
                    }
                    results.append(row)
                    trial_count += 1
                    if pred_idx == gt_idx:
                        running_correct += 1
                    running_acc = running_correct / len(results) if results else 0.0
                    progress.set_postfix(
                        {
                            "acc": f"{running_acc:.2%}",
                            "correct": f"{running_correct}/{len(results)}",
                            "top_k": top_k,
                            "eval_mode": eval_mode,
                            "frames": len(kept_specs),
                            "cand": candidate_frame_count,
                            "ret_sec": f"{retrieval_sec:.1f}",
                            "qwen_sec": f"{qwen_sec:.1f}",
                        }
                    )
                    progress.update(1)
                    if args.save_every > 0 and trial_count % args.save_every == 0:
                        save_outputs({"summary": summarize(results), "results": results}, args.output_path)
    finally:
        progress.close()
        payload = {"summary": summarize(results), "results": results}
        save_outputs(payload, args.output_path)
        print(f"Saved {len(results)} trials to {args.output_path}")
        print(f"Saved summary CSV to {Path(args.output_path).with_suffix('.summary.csv')}")
        print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
