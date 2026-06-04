#!/usr/bin/env python3
"""Run a minimal Qwen3-VL GPU smoke inference and exit."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor


def log(message: str) -> None:
    print(message, flush=True)


def make_test_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 420), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 55, 285, 305), fill=(62, 128, 235), outline=(20, 52, 112), width=5)
    draw.ellipse((350, 80, 570, 300), fill=(236, 101, 74), outline=(120, 40, 32), width=5)
    draw.line((80, 350, 560, 350), fill=(26, 26, 26), width=6)
    draw.text((72, 25), "Qwen3-VL GPU smoke test", fill=(20, 20, 20))
    draw.text((85, 320), "blue square", fill=(20, 20, 20))
    draw.text((395, 320), "orange circle", fill=(20, 20, 20))
    image.save(path)
    return path


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


def run_smoke(model_id: str, max_new_tokens: int, output_dir: Path) -> None:
    import transformers
    from qwen_vl_utils import process_vision_info

    log(f"hostname={socket.gethostname()}")
    log(f"python={sys.executable}")
    log(f"torch={torch.__version__}")
    log(f"transformers={transformers.__version__}")
    log(f"cuda_available={torch.cuda.is_available()}")
    log(f"cuda_device_count={torch.cuda.device_count()}")
    log(f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}")
    log(f"hf_home={os.getenv('HF_HOME', '')}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Slurm allocation.")

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        log(f"gpu[{idx}]={props.name}; total_memory_gb={props.total_memory / 1024**3:.1f}")

    image_path = make_test_image(output_dir / "qwen3vl_smoke_input.png")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": "In one short sentence, describe the main colored shapes in this image.",
                },
            ],
        }
    ]

    start = time.time()
    log(f"loading_processor={model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    log(f"loading_model={model_id}")
    model = load_model(model_id)
    model.eval()
    log(f"model_loaded_seconds={time.time() - start:.1f}")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    first_param = next(model.parameters())
    inputs = inputs.to(first_param.device)
    log(f"model_first_param_device={first_param.device}")

    gen_start = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated, strict=False)
    ]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    log(f"generation_seconds={time.time() - gen_start:.1f}")
    log(f"generated_answer={output_text}")
    if not output_text:
        raise RuntimeError("Model generated an empty answer.")
    log("QWEN3VL_SMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("hpc/smoke_outputs"))
    args = parser.parse_args()
    run_smoke(args.model_id, args.max_new_tokens, args.output_dir)


if __name__ == "__main__":
    main()
