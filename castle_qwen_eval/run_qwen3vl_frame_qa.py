#!/usr/bin/env python3
"""Run Qwen3-VL over official CASTLE/EgoVis questions and local frames."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MANIFEST = Path("outputs/castle_frame_eval/frame_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/castle_frame_eval")
OPTION_LETTERS = ("a", "b", "c", "d")


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


class QwenFrameGenerator:
    def __init__(self, model_id: str, max_new_tokens: int, max_image_pixels: int):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels
        start = time.time()
        log(f"loading_processor={model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        log(f"loading_model={model_id}")
        self.model = load_model(model_id)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        log(f"model_first_param_device={self.device}")
        log(f"model_loaded_seconds={time.time() - start:.1f}")

    def generate(self, prompt: str, image_paths: list[str]) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image_path, "max_pixels": self.max_image_pixels}
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
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


def normalize_prediction(pred: str) -> str:
    if not pred:
        return ""
    cleaned = pred.strip().lower()
    cleaned = cleaned.replace("the correct answer is", "")
    cleaned = cleaned.replace("answer:", "")
    cleaned = cleaned.replace("option", "")
    match = re.search(r"\b([abcd])\b", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"[abcd]", cleaned)
    return match.group(0) if match else ""


def question_prompt(question: dict[str, Any], source_label: str) -> str:
    answers = question["answers"]
    options = "\n".join(
        f"{letter.upper()}) {answers[letter]}" for letter in OPTION_LETTERS
    )
    return f"""You are answering an official CASTLE/EgoVis multiple-choice question using only the provided extracted video frames.

Evidence source condition: {source_label}

Question:
{question["query"]}

Options:
{options}

Answer only one letter: A, B, C, or D. Do not provide reasoning."""


def choose_frames(frames: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    indices = [
        round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)
    ]
    return [frames[i] for i in indices]


def choose_multi_frames(
    sources: list[dict[str, Any]],
    max_frames_per_source: int,
    max_total_frames: int,
) -> tuple[list[str], list[str]]:
    selected: list[tuple[str, str]] = []
    by_source = [
        (
            source["source_id"],
            choose_frames(source["frames"], max_frames_per_source),
        )
        for source in sources
    ]
    max_source_frames = max((len(frames) for _, frames in by_source), default=0)
    for frame_idx in range(max_source_frames):
        for source_id, frames in by_source:
            if frame_idx >= len(frames):
                continue
            selected.append((source_id, frames[frame_idx]))
            if max_total_frames > 0 and len(selected) >= max_total_frames:
                return [frame for _, frame in selected], [
                    source_id for source_id, _ in selected
                ]
    return [frame for _, frame in selected], [source_id for source_id, _ in selected]


def iter_eval_units(
    manifest: dict[str, Any],
    limit: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    question_by_id = {q["id"]: q for q in manifest["questions"]}
    units: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frame_set in manifest["frame_sets"]:
        for qid in frame_set.get("question_ids", []):
            question = question_by_id.get(qid)
            if question:
                units.append((frame_set, question))
                if limit is not None and len(units) >= limit:
                    return units
    return units


def run_single_sources(
    generator: QwenFrameGenerator,
    frame_set: dict[str, Any],
    question: dict[str, Any],
    max_frames_per_source: int,
    predictions_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for source in frame_set["sources"]:
        image_paths = choose_frames(source["frames"], max_frames_per_source)
        source_label = (
            f"single source {source['source_id']} ({source.get('source_type', 'unknown')})"
        )
        pred = generator.generate(question_prompt(question, source_label), image_paths)
        pred_letter = normalize_prediction(pred)
        row = {
            "dataset": "castle_egovis_official_frames",
            "metric_type": "answer_difference_only",
            "has_answer_key": False,
            "frame_set_id": frame_set["frame_set_id"],
            "window_id": frame_set.get("window_id", ""),
            "mapping_status": frame_set.get("mapping_status", ""),
            "question_id": question["id"],
            "question": question["query"],
            "answers": question["answers"],
            "condition": "single",
            "source_ids": [source["source_id"]],
            "source_types": [source.get("source_type", "unknown")],
            "num_images": len(image_paths),
            "image_paths": image_paths,
            "raw_prediction": pred,
            "prediction": pred_letter,
        }
        append_jsonl(predictions_path, row)
        rows.append(row)
        log(
            f"single frame_set={frame_set['frame_set_id']} qid={question['id']} "
            f"source={source['source_id']} pred={pred_letter or 'INVALID'}"
        )
    return rows


def run_multi_source(
    generator: QwenFrameGenerator,
    frame_set: dict[str, Any],
    question: dict[str, Any],
    max_frames_per_source: int,
    max_total_frames: int,
    predictions_path: Path,
) -> dict[str, Any]:
    source_ids = []
    source_types = []
    for source in frame_set["sources"]:
        source_ids.append(source["source_id"])
        source_types.append(source.get("source_type", "unknown"))
    image_paths, image_source_ids = choose_multi_frames(
        frame_set["sources"],
        max_frames_per_source,
        max_total_frames,
    )
    source_label = "multi source " + ", ".join(source_ids)
    pred = generator.generate(question_prompt(question, source_label), image_paths)
    pred_letter = normalize_prediction(pred)
    row = {
        "dataset": "castle_egovis_official_frames",
        "metric_type": "answer_difference_only",
        "has_answer_key": False,
        "frame_set_id": frame_set["frame_set_id"],
        "window_id": frame_set.get("window_id", ""),
        "mapping_status": frame_set.get("mapping_status", ""),
        "question_id": question["id"],
        "question": question["query"],
        "answers": question["answers"],
        "condition": "multi",
        "source_ids": source_ids,
        "source_types": source_types,
        "image_source_ids": image_source_ids,
        "num_images": len(image_paths),
        "image_paths": image_paths,
        "raw_prediction": pred,
        "prediction": pred_letter,
    }
    append_jsonl(predictions_path, row)
    log(
        f"multi frame_set={frame_set['frame_set_id']} qid={question['id']} "
        f"sources={len(source_ids)} pred={pred_letter or 'INVALID'}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames-per-source", type=int, default=1)
    parser.add_argument("--max-total-multi-frames", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument(
        "--condition",
        choices=["single", "multi", "all"],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite:
        raise FileExistsError(f"{predictions_path} exists; use --overwrite")
    if predictions_path.exists():
        predictions_path.unlink()

    log(f"hostname={socket.gethostname()}")
    log(f"python={sys.executable}")
    log(f"cuda_available={torch.cuda.is_available()}")
    log(f"cuda_device_count={torch.cuda.device_count()}")
    log(f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this in a GPU Slurm allocation.")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        log(f"gpu[{idx}]={props.name}; total_memory_gb={props.total_memory / 1024**3:.1f}")

    manifest = load_json(args.manifest)
    units = iter_eval_units(manifest, args.limit)
    log(f"eval_units={len(units)}")
    if not units:
        raise SystemExit("No official questions could be mapped to available frames.")

    generator = QwenFrameGenerator(
        args.model_id,
        args.max_new_tokens,
        args.max_image_pixels,
    )
    for frame_set, question in units:
        if args.condition in {"single", "all"}:
            run_single_sources(
                generator,
                frame_set,
                question,
                args.max_frames_per_source,
                predictions_path,
            )
        if args.condition in {"multi", "all"}:
            run_multi_source(
                generator,
                frame_set,
                question,
                args.max_frames_per_source,
                args.max_total_multi_frames,
                predictions_path,
            )

    log(f"saved_predictions={predictions_path}")


if __name__ == "__main__":
    main()
