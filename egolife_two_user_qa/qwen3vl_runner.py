"""Qwen3-VL runner backends for local/open-source inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_IMAGE_PIXELS = 262144


class Generator(Protocol):
    model_id: str

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        ...


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def file_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_transformers_model(model_id: str, dtype: str = "bfloat16"):
    try:
        import torch
        from transformers import AutoModelForImageTextToText

        try:
            from transformers import Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        except Exception:
            model_cls = AutoModelForImageTextToText

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
        try:
            return model_cls.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
        except (TypeError, ValueError):
            return model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    except ImportError as exc:
        raise RuntimeError(
            "transformers-local backend requires torch, transformers>=4.57, and qwen-vl-utils"
        ) from exc


class Qwen3VLTransformersRunner:
    """Run Qwen3-VL directly through Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        max_new_tokens: int = 1024,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        dtype: str = "bfloat16",
        allow_cpu: bool = False,
    ) -> None:
        if not allow_cpu and not cuda_available():
            raise RuntimeError(
                "CUDA is not available. Use --dry-run, --backend openai-compatible-local, "
                "or pass allow_cpu=True only for tiny tests."
            )
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels
        self.process_vision_info = process_vision_info
        start = time.time()
        print(f"loading_processor={model_id}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"loading_model={model_id}", flush=True)
        self.model = load_transformers_model(model_id, dtype=dtype)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.torch = torch
        print(f"model_first_param_device={self.device}", flush=True)
        print(f"model_loaded_seconds={time.time() - start:.1f}", flush=True)

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        video_paths = video_paths or []
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image_path, "max_pixels": self.max_image_pixels}
            for image_path in image_paths
        ]
        content.extend(
            {"type": "video", "video": video_path, "max_pixels": self.max_image_pixels, "fps": 1.0}
            for video_path in video_paths
        )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            vision_info = self.process_vision_info(messages, return_video_kwargs=True)
            image_inputs, video_inputs, video_kwargs = vision_info
        except TypeError:
            image_inputs, video_inputs = self.process_vision_info(messages)
            video_kwargs = {}
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class OpenAICompatibleLocalRunner:
    """Call a local vLLM/SGLang/llama.cpp OpenAI-compatible server."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        max_new_tokens: int = 1024,
        timeout: int = 600,
        api_key: str | None = None,
        allow_video_input: bool = False,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.api_key = api_key or os.getenv("LOCAL_VLM_API_KEY") or "none"
        self.allow_video_input = allow_video_input

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
        if video_paths and not self.allow_video_input:
            raise RuntimeError(
                "openai-compatible-local backend received video_paths, but video input is disabled. "
                "Use image fallback or pass --allow-openai-video-input for a server that supports video data URLs."
            )
        for path in video_paths or []:
            content.append({"type": "video_url", "video_url": {"url": file_to_data_url(path)}})
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


class DryRunRunner:
    """A no-model runner used only to write prompts and test plumbing."""

    model_id = "dry-run-no-model"

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        return json.dumps(
            {
                "dry_run": True,
                "prompt_preview": prompt[:1000],
                "image_count": len(image_paths or []),
                "video_count": len(video_paths or []),
            },
            ensure_ascii=False,
        )


def make_runner(
    backend: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_new_tokens: int = 1024,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
) -> Generator:
    if backend == "transformers-local":
        return Qwen3VLTransformersRunner(
            model_id,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
        )
    if backend == "openai-compatible-local":
        return OpenAICompatibleLocalRunner(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            allow_video_input=allow_openai_video_input,
        )
    if backend == "dry-run":
        return DryRunRunner()
    raise ValueError(f"Unsupported backend: {backend}")
