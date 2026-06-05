"""
Evaluate DAY1 MA-EgoQA questions with Qwen3-VL using raw MaEgo video frames.

The script compares three visual-context settings:
  1. single: each relevant agent separately
  2. pair: every pair of relevant agents
  3. all: all relevant agents together

Frame modes 5, 10, and 15 are per-agent frame budgets sampled uniformly across each selected agent's clips/timeframes.
  --frame-modes all -> every decoded frame, optionally thinned by --all-frame-stride
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from egomas.utils.io import load_json, save_json
from egomas.utils.parsing import get_prediction_index


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
BENCHMARK_PATH = "data/MA-EgoQA.json"
VIDEO_ROOT = "/scratch/jz3420/data/MaEgo"
OUTPUT_PATH = "results_day1_qwen3vl_frames.json"

AGENT_TO_FOLDER = {
    "JAKE": "A1_JAKE",
    "ALICE": "A2_ALICE",
    "TASHA": "A3_TASHA",
    "LUCIA": "A4_LUCIA",
    "KATRINA": "A5_KATRINA",
    "SHURE": "A6_SHURE",
}

CONTEXT_RE = re.compile(r"^(DAY\d+)_(\d{8})_(\d{8})$")
VIDEO_RE = re.compile(r"^(DAY\d+)_([A-Z0-9]+_[A-Z]+)_(\d{8})\.mp4$")


@dataclass(frozen=True)
class ContextWindow:
    key: str
    day: str
    start_token: str
    end_token: str
    start_sec: float
    end_sec: float
    agents: tuple[str, ...]


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    start_token: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class FrameSpec:
    agent: str
    context_key: str
    video_path: str
    video_start_token: str
    offset_sec: float
    timestamp_sec: float


def parse_time_token(token: str) -> float:
    """Parse HHMMSScc tokens used by the MaEgo filenames/context keys."""
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"Invalid time token: {token}")
    hh = int(token[0:2])
    mm = int(token[2:4])
    ss = int(token[4:6])
    centisec = int(token[6:8])
    return hh * 3600 + mm * 60 + ss + centisec / 100.0


def normalize_agent(name: str) -> str:
    return name.strip().upper()


def parse_contexts(item: dict[str, Any]) -> list[ContextWindow]:
    windows: list[ContextWindow] = []
    contexts = item.get("contexts", {})
    if not isinstance(contexts, dict):
        return windows
    for key, agents in contexts.items():
        match = CONTEXT_RE.match(key)
        if not match:
            continue
        day, start_token, end_token = match.groups()
        if day != "DAY1":
            continue
        norm_agents = tuple(normalize_agent(agent) for agent in agents)
        windows.append(
            ContextWindow(
                key=key,
                day=day,
                start_token=start_token,
                end_token=end_token,
                start_sec=parse_time_token(start_token),
                end_sec=parse_time_token(end_token),
                agents=norm_agents,
            )
        )
    return windows


def is_day1_item(item: dict[str, Any]) -> bool:
    return bool(parse_contexts(item))


def relevant_agents(windows: Iterable[ContextWindow]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for window in windows:
        for agent in window.agents:
            if agent in AGENT_TO_FOLDER and agent not in seen:
                seen.add(agent)
                ordered.append(agent)
    return ordered


def build_video_index(video_root: str | Path, day: str = "DAY1") -> dict[str, list[VideoSegment]]:
    root = Path(video_root)
    index: dict[str, list[VideoSegment]] = {}
    for agent, folder in AGENT_TO_FOLDER.items():
        day_dir = root / folder / day
        segments: list[VideoSegment] = []
        for path in sorted(day_dir.glob("*.mp4")):
            match = VIDEO_RE.match(path.name)
            if not match:
                continue
            file_day, _, start_token = match.groups()
            if file_day != day:
                continue
            start_sec = parse_time_token(start_token)
            segments.append(
                VideoSegment(
                    path=path,
                    start_token=start_token,
                    start_sec=start_sec,
                    end_sec=start_sec + 30.0,
                )
            )
        index[agent] = segments
    return index


def overlapping_segments(
    index: dict[str, list[VideoSegment]], agent: str, window: ContextWindow
) -> list[VideoSegment]:
    segments = index.get(agent, [])
    return [
        segment
        for segment in segments
        if segment.start_sec < window.end_sec and segment.end_sec > window.start_sec
    ]


def frame_budget(frame_mode: str) -> int | None:
    if frame_mode == "all":
        return None
    try:
        budget = int(frame_mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported frame mode: {frame_mode}") from exc
    if budget <= 0:
        raise ValueError(f"Frame budget must be positive: {frame_mode}")
    return budget


def safe_offset(offset: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    # Seeking exactly at the segment end often lands beyond the last frame.
    return min(max(offset, 0.0), max(duration - 0.05, 0.0))


def load_frame_dependencies():
    try:
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing frame extraction dependencies. Install them with:\n"
            "  python -m pip install pillow opencv-python-headless\n"
            f"Original error: {exc}"
        ) from exc
    return cv2, Image


def decode_frame_indices(
    cap: Any,
    cv2: Any,
    Image: Any,
    segment: VideoSegment,
    window: ContextWindow,
    agent: str,
    fps: float,
    frame_indices: list[int],
) -> tuple[list[Any], list[FrameSpec]]:
    images: list[Any] = []
    specs: list[FrameSpec] = []
    wanted = sorted(set(frame_indices))
    wanted_set = set(wanted)
    max_idx = wanted[-1] if wanted else -1
    current_idx = 0
    while current_idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if current_idx in wanted_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            offset_sec = current_idx / fps if fps > 0 else 0.0
            images.append(image)
            specs.append(
                FrameSpec(
                    agent=agent,
                    context_key=window.key,
                    video_path=str(segment.path),
                    video_start_token=segment.start_token,
                    offset_sec=round(offset_sec, 3),
                    timestamp_sec=round(segment.start_sec + offset_sec, 3),
                )
            )
            if len(images) == len(wanted):
                break
        current_idx += 1
    return images, specs


def extract_frames_from_segment(
    segment: VideoSegment,
    window: ContextWindow,
    agent: str,
    frame_mode: str,
    all_frame_stride: int,
    max_frames_per_segment: int,
    sample_offsets: list[float] | None = None,
) -> tuple[list[Any], list[FrameSpec]]:
    cv2, Image = load_frame_dependencies()
    cap = cv2.VideoCapture(str(segment.path))
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 30.0
    context_start_offset = max(window.start_sec - segment.start_sec, 0.0)
    context_end_offset = min(window.end_sec - segment.start_sec, duration)

    frame_indices: list[int] = []
    if sample_offsets is not None:
        for offset in sample_offsets:
            local_offset = safe_offset(offset, duration)
            if local_offset < context_start_offset or local_offset > context_end_offset:
                continue
            idx = int(round(local_offset * fps))
            if frame_count > 0:
                idx = min(idx, frame_count - 1)
            frame_indices.append(max(idx, 0))
    elif frame_mode == "all":
        start_idx = max(int(math.floor(context_start_offset * fps)), 0)
        end_idx = min(int(math.ceil(context_end_offset * fps)), max(frame_count - 1, 0))
        stride = max(all_frame_stride, 1)
        frame_indices = list(range(start_idx, end_idx + 1, stride))
    else:
        raise ValueError(
            "Numeric frame modes are total trial budgets and must be sampled in collect_frames."
        )

    if max_frames_per_segment > 0:
        frame_indices = frame_indices[:max_frames_per_segment]

    images, specs = decode_frame_indices(
        cap=cap,
        cv2=cv2,
        Image=Image,
        segment=segment,
        window=window,
        agent=agent,
        fps=fps,
        frame_indices=frame_indices,
    )
    cap.release()
    return images, specs


def collect_frame_spans(
    video_index: dict[str, list[VideoSegment]],
    windows: list[ContextWindow],
    agents: tuple[str, ...],
) -> list[tuple[ContextWindow, str, VideoSegment, float, float, float]]:
    spans: list[tuple[ContextWindow, str, VideoSegment, float, float, float]] = []
    for window in windows:
        for agent in agents:
            if agent not in window.agents:
                continue
            for segment in overlapping_segments(video_index, agent, window):
                local_start = max(window.start_sec - segment.start_sec, 0.0)
                local_end = min(window.end_sec - segment.start_sec, segment.end_sec - segment.start_sec)
                duration = max(local_end - local_start, 0.0)
                if duration > 0:
                    spans.append((window, agent, segment, local_start, local_end, duration))
    return spans


def allocate_uniform_offsets(
    spans: list[tuple[ContextWindow, str, VideoSegment, float, float, float]],
    budget: int,
) -> dict[tuple[str, str, str], list[float]]:
    total_duration = sum(span[-1] for span in spans)
    if budget <= 0 or total_duration <= 0:
        return {}

    positions = [((idx + 0.5) * total_duration / budget) for idx in range(budget)]
    allocated: dict[tuple[str, str, str], list[float]] = {}
    span_idx = 0
    elapsed_before_span = 0.0
    for position in positions:
        while span_idx < len(spans) - 1 and position >= elapsed_before_span + spans[span_idx][-1]:
            elapsed_before_span += spans[span_idx][-1]
            span_idx += 1
        window, agent, segment, local_start, local_end, duration = spans[span_idx]
        offset = local_start + (position - elapsed_before_span)
        offset = min(max(offset, local_start), local_end)
        key = (window.key, agent, str(segment.path))
        allocated.setdefault(key, []).append(offset)
    return allocated


def collect_frames(
    video_index: dict[str, list[VideoSegment]],
    windows: list[ContextWindow],
    agents: tuple[str, ...],
    frame_mode: str,
    all_frame_stride: int,
    max_frames_per_segment: int,
) -> tuple[list[Any], list[FrameSpec]]:
    images: list[Any] = []
    specs: list[FrameSpec] = []
    budget = frame_budget(frame_mode)
    spans = collect_frame_spans(video_index, windows, agents)

    if budget is not None:
        for selected_agent in agents:
            agent_spans = [span for span in spans if span[1] == selected_agent]
            allocated = allocate_uniform_offsets(agent_spans, budget)
            for window, agent, segment, _, _, _ in agent_spans:
                key = (window.key, agent, str(segment.path))
                offsets = allocated.get(key)
                if not offsets:
                    continue
                seg_images, seg_specs = extract_frames_from_segment(
                    segment=segment,
                    window=window,
                    agent=agent,
                    frame_mode=frame_mode,
                    all_frame_stride=all_frame_stride,
                    max_frames_per_segment=max_frames_per_segment,
                    sample_offsets=offsets,
                )
                images.extend(seg_images)
                specs.extend(seg_specs)
        return images, specs

    for window, agent, segment, _, _, _ in spans:
        seg_images, seg_specs = extract_frames_from_segment(
            segment=segment,
            window=window,
            agent=agent,
            frame_mode=frame_mode,
            all_frame_stride=all_frame_stride,
            max_frames_per_segment=max_frames_per_segment,
        )
        images.extend(seg_images)
        specs.extend(seg_specs)
    return images, specs


def build_question_prompt(item: dict[str, Any], agents: tuple[str, ...], frame_specs: list[FrameSpec]) -> str:
    options = item["options"]
    option_text = "\n".join(
        f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)
    )
    labels = "\n".join(
        f"Frame {idx + 1}: agent={spec.agent}, context={spec.context_key}, "
        f"video={Path(spec.video_path).name}, offset={spec.offset_sec:.2f}s"
        for idx, spec in enumerate(frame_specs)
    )
    return (
        "You are answering a multiple-choice egocentric video QA question.\n"
        "Use only the provided video frames. If the frames are insufficient, choose the best supported option.\n"
        f"Agents included in this trial: {', '.join(agents)}\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Frame metadata in image order:\n{labels}\n\n"
        "Return exactly one option letter: A, B, C, D, or E."
    )


def load_qwen_model(model_name: str, device_map: str, min_pixels: int, max_pixels: int):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing Qwen3-VL dependencies. Install them with something like:\n"
            "  python -m pip install torch torchvision transformers accelerate pillow opencv-python-headless\n"
            "Depending on the cluster CUDA setup, you may need the HPC's recommended PyTorch install command.\n"
            f"Original error: {exc}"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor, torch


def run_qwen_answer(
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str,
    images: list[Any],
    max_new_tokens: int,
) -> str:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (decoded[0] if decoded else "").strip()


def trial_groups(agents: list[str], eval_modes: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    groups: list[tuple[str, tuple[str, ...]]] = []
    if "single" in eval_modes:
        groups.extend(("single", (agent,)) for agent in agents)
    if "pair" in eval_modes:
        groups.extend(("pair", pair) for pair in itertools.combinations(agents, 2))
    if "all" in eval_modes:
        groups.append(("all", tuple(agents)))
    return groups


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_frame_mode": {}, "by_eval_mode": {}}
    if not results:
        return summary

    def acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answered = [row for row in rows if row.get("pred_index", -1) >= 0]
        correct = sum(1 for row in rows if row.get("correct"))
        return {
            "trials": len(rows),
            "answered": len(answered),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "avg_seconds": sum(row.get("elapsed_sec", 0.0) for row in rows) / len(rows),
        }

    summary["overall"] = acc(results)
    for key in sorted({row["frame_mode"] for row in results}):
        summary["by_frame_mode"][key] = acc([row for row in results if row["frame_mode"] == key])
    for key in sorted({row["eval_mode"] for row in results}):
        summary["by_eval_mode"][key] = acc([row for row in results if row["eval_mode"] == key])

    single_rows = [
        row
        for row in results
        if row.get("eval_mode") == "single" and len(row.get("agents", [])) == 1
    ]
    if single_rows:
        summary["by_single_agent"] = {}
        for agent in sorted({row["agents"][0] for row in single_rows}):
            agent_rows = [row for row in single_rows if row["agents"][0] == agent]
            summary["by_single_agent"][agent] = acc(agent_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH)
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--frame-modes", nargs="+", default=["5"], choices=["5", "10", "15", "all"])
    parser.add_argument("--eval-modes", nargs="+", default=["single", "pair", "all"], choices=["single", "pair", "all"])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of DAY1 questions; 0 means all.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset within filtered DAY1 questions.")
    parser.add_argument("--all-frame-stride", type=int, default=1, help="Use every Nth decoded frame for --frame-modes all.")
    parser.add_argument("--max-frames-per-segment", type=int, default=0, help="Safety cap per 30s segment; 0 means no cap.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Load data and extract frames, but do not load/run Qwen.")
    args = parser.parse_args()

    benchmark = load_json(args.benchmark_path)
    day1_items = [item for item in benchmark if is_day1_item(item)]
    if args.start_index:
        day1_items = day1_items[args.start_index :]
    if args.limit > 0:
        day1_items = day1_items[: args.limit]

    print(f"Loaded {len(benchmark)} benchmark items; evaluating {len(day1_items)} DAY1 items.")
    print(f"Frame modes: {args.frame_modes}; eval modes: {args.eval_modes}")

    video_index = build_video_index(args.video_root, day="DAY1")
    for agent, segments in video_index.items():
        print(f"{agent}: {len(segments)} DAY1 video segments")

    model = processor = torch = None
    if not args.dry_run:
        model, processor, torch = load_qwen_model(
            args.model_name,
            args.device_map,
            args.min_pixels,
            args.max_pixels,
        )

    results: list[dict[str, Any]] = []
    running_correct = 0
    running_answered = 0
    single_agent_stats: dict[str, dict[str, int]] = {}
    total_trials = 0
    for item in day1_items:
        agents = relevant_agents(parse_contexts(item))
        total_trials += len(args.frame_modes) * len(trial_groups(agents, args.eval_modes))

    progress = tqdm(total=total_trials, desc="DAY1 Qwen3-VL eval")
    trial_count = 0
    try:
        for q_idx, item in enumerate(day1_items):
            windows = parse_contexts(item)
            agents = relevant_agents(windows)
            gt_idx = item["options"].index(item["answer"])
            groups = trial_groups(agents, args.eval_modes)

            for frame_mode in args.frame_modes:
                for eval_mode, group_agents in groups:
                    started = time.perf_counter()
                    images, specs = collect_frames(
                        video_index=video_index,
                        windows=windows,
                        agents=group_agents,
                        frame_mode=frame_mode,
                        all_frame_stride=args.all_frame_stride,
                        max_frames_per_segment=args.max_frames_per_segment,
                    )
                    prompt = build_question_prompt(item, group_agents, specs)
                    if args.dry_run:
                        raw_pred = "DRY_RUN"
                        pred_idx = -1
                    elif not images:
                        raw_pred = "NO_FRAMES"
                        pred_idx = -1
                    else:
                        raw_pred = run_qwen_answer(
                            model=model,
                            processor=processor,
                            torch=torch,
                            prompt=prompt,
                            images=images,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred_idx = get_prediction_index(raw_pred)
                    elapsed = time.perf_counter() - started
                    row = {
                        "question_index": args.start_index + q_idx,
                        "category": item.get("category"),
                        "subcategory": item.get("subcategory"),
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"],
                        "gt_index": gt_idx,
                        "contexts": item.get("contexts", {}),
                        "frame_mode": frame_mode,
                        "eval_mode": eval_mode,
                        "agents": list(group_agents),
                        "num_frames": len(images),
                        "frames": [spec.__dict__ for spec in specs],
                        "raw_pred": raw_pred,
                        "pred_index": pred_idx,
                        "correct": pred_idx == gt_idx,
                        "elapsed_sec": elapsed,
                    }
                    results.append(row)
                    trial_count += 1
                    if pred_idx >= 0:
                        running_answered += 1
                    if pred_idx == gt_idx:
                        running_correct += 1
                    if eval_mode == "single" and len(group_agents) == 1:
                        agent = group_agents[0]
                        stats = single_agent_stats.setdefault(agent, {"trials": 0, "correct": 0})
                        stats["trials"] += 1
                        if pred_idx == gt_idx:
                            stats["correct"] += 1
                    running_acc = running_correct / len(results) if results else 0.0
                    answered_acc = (
                        running_correct / running_answered if running_answered else 0.0
                    )
                    avg_sec = sum(r["elapsed_sec"] for r in results) / len(results)
                    postfix = {
                        "acc": f"{running_acc:.2%}",
                        "correct": f"{running_correct}/{len(results)}",
                        "answered_acc": f"{answered_acc:.2%}",
                        "frame_mode": frame_mode,
                        "eval_mode": eval_mode,
                        "frames": len(images),
                        "sec": f"{elapsed:.1f}",
                        "avg_sec": f"{avg_sec:.1f}",
                    }
                    if single_agent_stats:
                        parts = []
                        for agent in sorted(single_agent_stats):
                            stats = single_agent_stats[agent]
                            trials = stats["trials"]
                            correct = stats["correct"]
                            acc = correct / trials if trials else 0.0
                            parts.append(f"{agent}:{correct}/{trials} {acc:.1%}")
                        postfix["single_acc"] = " | ".join(parts)
                    progress.set_postfix(postfix)
                    progress.update(1)

                    if args.save_every > 0 and trial_count % args.save_every == 0:
                        save_json({"summary": summarize(results), "results": results}, args.output_path)
    finally:
        progress.close()
        save_json({"summary": summarize(results), "results": results}, args.output_path)
        print(f"Saved {len(results)} trials to {args.output_path}")
        print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
