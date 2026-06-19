"""Vision-language runner backends for local and API inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-pro"
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


def normalize_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep Qwen video kwargs compatible across qwen-vl-utils/Transformers versions."""

    normalized = dict(video_kwargs)
    if isinstance(normalized.get("fps"), list):
        fps_values = normalized["fps"]
        normalized["fps"] = fps_values[0] if fps_values else 1.0
    return normalized


def coerce_video_metadata(value: Any) -> Any:
    """Return a Transformers-compatible video metadata object when possible."""

    if not isinstance(value, dict):
        return value
    frames_indices = value.get("frames_indices")
    if frames_indices is not None:
        frames_indices = list(frames_indices)
    total_num_frames = value.get("total_num_frames")
    if total_num_frames is None and frames_indices is not None:
        total_num_frames = len(frames_indices)
    try:
        total_num_frames = int(round(float(total_num_frames)))
    except (TypeError, ValueError):
        total_num_frames = 0
    kwargs = {
        "total_num_frames": total_num_frames,
        "fps": value.get("fps"),
        "width": value.get("width"),
        "height": value.get("height"),
        "duration": value.get("duration"),
        "video_backend": value.get("video_backend"),
        "frames_indices": frames_indices,
    }
    try:
        from transformers.video_utils import VideoMetadata

        return VideoMetadata(**kwargs)
    except Exception:
        return SimpleNamespace(**kwargs)


def split_video_inputs_and_metadata(
    video_inputs: Any,
    video_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Split qwen-vl-utils ``(video, metadata)`` pairs for Qwen3-VL processors."""

    if video_inputs is None:
        return video_inputs, normalize_video_kwargs(video_kwargs)
    normalized_kwargs = normalize_video_kwargs(video_kwargs)
    fixed_video_inputs = []
    metadata_rows = []
    found_metadata = False
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            video, metadata = item
            fixed_video_inputs.append(video)
            metadata_rows.append(coerce_video_metadata(metadata))
            found_metadata = True
        else:
            fixed_video_inputs.append(item)
            metadata_rows.append(None)
    if found_metadata:
        normalized_kwargs["video_metadata"] = metadata_rows
        normalized_kwargs["return_metadata"] = True
    return fixed_video_inputs, normalized_kwargs


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
        start = time.time()
        print(
            "qwen_generate_start "
            f"images={len(image_paths)} videos={len(video_paths)} prompt_chars={len(prompt)}",
            flush=True,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            vision_info = self.process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            image_inputs, video_inputs, video_kwargs = vision_info
        except TypeError:
            try:
                image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                    messages,
                    return_video_kwargs=True,
                )
            except TypeError:
                image_inputs, video_inputs = self.process_vision_info(messages)
                video_kwargs = {}
        vision_seconds = time.time() - start
        print(f"qwen_vision_processed_seconds={vision_seconds:.1f}", flush=True)
        video_inputs, video_kwargs = split_video_inputs_and_metadata(video_inputs, video_kwargs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)
        encode_seconds = time.time() - start
        input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
        inputs.pop("video_metadata", None)
        print(
            f"qwen_processor_encoded_seconds={encode_seconds:.1f} input_tokens={input_tokens}",
            flush=True,
        )
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        total_seconds = time.time() - start
        output_tokens = int(generated.shape[-1] - inputs.input_ids.shape[-1])
        print(
            f"qwen_model_generate_seconds={total_seconds - encode_seconds:.1f} "
            f"total_seconds={total_seconds:.1f} output_tokens={output_tokens}",
            flush=True,
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


class GeminiAPIRunner:
    """Call the Gemini API through REST using only the Python standard library."""

    def __init__(
        self,
        model_id: str = DEFAULT_GEMINI_MODEL_ID,
        *,
        max_new_tokens: int = 1024,
        upload_poll_seconds: float = 2.0,
        upload_timeout_seconds: float = 300.0,
        timeout: int = 600,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "gemini-api backend requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
            )
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.upload_poll_seconds = upload_poll_seconds
        self.upload_timeout_seconds = upload_timeout_seconds
        self.timeout = timeout
        self.api_key = api_key
        self.api_base_url = os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        self.upload_base_url = os.getenv(
            "GEMINI_UPLOAD_BASE_URL",
            "https://generativelanguage.googleapis.com/upload/v1beta",
        ).rstrip("/")
        self._file_cache: dict[str, dict[str, Any]] = {}
        self.last_usage_metadata: dict[str, Any] = {}

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        method: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"x-goog-api-key": self.api_key}
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read().decode("utf-8")
                return (json.loads(data) if data else {}, resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code} for {url}: {detail}") from exc

    def _upload_file(self, path: str) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        start_payload = {"file": {"display_name": Path(path).name}}
        _, start_resp = self._request_json(
            f"{self.upload_base_url}/files",
            payload=start_payload,
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
            },
            method="POST",
        )
        upload_url = start_resp.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Gemini upload did not return X-Goog-Upload-URL")
        with open(path, "rb") as handle:
            upload_data = handle.read()
        upload_req = urllib.request.Request(
            upload_url,
            data=upload_data,
            headers={
                "Content-Length": str(size),
                "Content-Type": mime_type,
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(upload_req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini file upload HTTP {exc.code} for {path}: {detail}") from exc
        file_info = response.get("file", response)
        file_info.setdefault("mimeType", mime_type)
        return self._wait_for_file(file_info)

    def _wait_for_file(self, file_info: dict[str, Any]) -> dict[str, Any]:
        name = file_info.get("name")
        if not name:
            return file_info
        start = time.time()
        current = file_info
        while True:
            state = str(current.get("state", "")).upper()
            if "FAILED" in state:
                raise RuntimeError(f"Gemini file processing failed for {name}: {current!r}")
            if not state or "PROCESSING" not in state:
                return current
            if time.time() - start > self.upload_timeout_seconds:
                raise TimeoutError(f"Timed out waiting for Gemini file processing: {name}")
            time.sleep(self.upload_poll_seconds)
            current, _ = self._request_json(f"{self.api_base_url}/{name}", method="GET")

    def _uploaded_file_part(self, path: str | Path) -> dict[str, Any]:
        path = str(Path(path).resolve())
        if path not in self._file_cache:
            mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            print(f"gemini_upload_start path={path} mime_type={mime_type}", flush=True)
            uploaded = self._upload_file(path)
            print(
                "gemini_upload_done "
                f"name={uploaded.get('name', '')} uri={uploaded.get('uri', '')}",
                flush=True,
            )
            self._file_cache[path] = uploaded
        uploaded = self._file_cache[path]
        mime_type = uploaded.get("mimeType") or uploaded.get("mime_type") or mimetypes.guess_type(path)[0]
        return {"file_data": {"file_uri": uploaded["uri"], "mime_type": mime_type}}

    def _inline_file_part(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"inline_data": {"data": data, "mime_type": mime_type}}

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        video_paths = video_paths or []
        parts = []
        parts.extend(self._inline_file_part(path) for path in image_paths)
        parts.extend(self._uploaded_file_part(path) for path in video_paths)
        parts.append({"text": prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_new_tokens,
                "responseMimeType": "application/json",
            },
        }
        model_name = self.model_id if self.model_id.startswith("models/") else f"models/{self.model_id}"
        start = time.time()
        print(
            "gemini_generate_start "
            f"model={self.model_id} images={len(image_paths)} videos={len(video_paths)} "
            f"prompt_chars={len(prompt)}",
            flush=True,
        )
        response, _ = self._request_json(
            f"{self.api_base_url}/{model_name}:generateContent",
            payload=payload,
            method="POST",
        )
        self.last_usage_metadata = response.get("usageMetadata", {})
        usage = json.dumps(self.last_usage_metadata, ensure_ascii=False, sort_keys=True)
        print(f"gemini_generate_done seconds={time.time() - start:.1f} usage={usage}", flush=True)
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini response did not contain text: {response!r}") from exc


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
    if backend == "gemini-api" and model_id == DEFAULT_MODEL_ID:
        model_id = DEFAULT_GEMINI_MODEL_ID
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
    if backend == "gemini-api":
        return GeminiAPIRunner(
            model_id,
            max_new_tokens=max_new_tokens,
        )
    if backend == "dry-run":
        return DryRunRunner()
    raise ValueError(f"Unsupported backend: {backend}")
