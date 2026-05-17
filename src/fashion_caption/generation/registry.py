from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional

from PIL import Image

from fashion_caption.generation.base import GenerationResult
from fashion_caption.generation.openai_api import (
    OpenAIConfigurationError,
    OpenAIGenerationError,
    OpenAIResponsesClient,
    openai_key_configured,
)
from fashion_caption.postprocess.text import clean_description
from fashion_caption.prompts import PromptConfig


LOCAL_MODEL_IDS = {"vit-gpt2", "blip", "blip2", "blip2-lora"}
REMOTE_MODEL_IDS = {"blip2", "blip2-lora"}
ALL_MODEL_IDS = {"vit-gpt2", "blip", "blip2", "blip2-lora", "gpt"}


def auto_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _max_new_tokens(params: Dict[str, Any], default: int = 60) -> int:
    value = params.get("max_new_tokens", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _max_words(params: Dict[str, Any]) -> int:
    value = params.get("max_words", 25)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 25


def _clean_result(raw_text: str, prompt_config: PromptConfig, params: Dict[str, Any]) -> str:
    return clean_description(
        raw_text,
        prompt=prompt_config.prompt,
        article_type=prompt_config.article_type,
        base_colour=prompt_config.base_colour,
        max_words=_max_words(params),
    )


class VitGpt2Generator:
    model_id = "vit-gpt2"
    instruction_capable = False

    def __init__(self, device=None):
        self.device = device or auto_device()
        self._bundle = None

    def _load(self):
        if self._bundle is None:
            from fashion_caption.models import vitgpt2

            self._bundle = vitgpt2.load_model(self.device)
        return self._bundle

    def generate(self, image: Image.Image, prompt_config: PromptConfig, params: Dict[str, Any]) -> GenerationResult:
        from fashion_caption.models import vitgpt2

        processor, tokenizer, model = self._load()
        started = time.perf_counter()
        result = vitgpt2.generate_text_details(
            image=image,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=_max_new_tokens(params, 40),
            num_beams=int(params.get("num_beams", 3)),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _clean_result(result["description"], prompt_config, params)
        return GenerationResult(
            text=text,
            meta={
                "model_id": self.model_id,
                "instruction_capable": self.instruction_capable,
                "device": str(self.device),
                "raw_output": result["raw_output"],
                "latency_ms": latency_ms,
            },
        )


class BlipGenerator:
    model_id = "blip"
    instruction_capable = False

    def __init__(self, device=None):
        self.device = device or auto_device()
        self._bundle = None

    def _load(self):
        if self._bundle is None:
            from fashion_caption.models import blip

            self._bundle = blip.load_model(self.device)
        return self._bundle

    def generate(self, image: Image.Image, prompt_config: PromptConfig, params: Dict[str, Any]) -> GenerationResult:
        from fashion_caption.models import blip

        processor, model = self._load()
        started = time.perf_counter()
        result = blip.generate_text_details(
            image=image,
            processor=processor,
            model=model,
            max_new_tokens=_max_new_tokens(params, 40),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _clean_result(result["description"], prompt_config, params)
        return GenerationResult(
            text=text,
            meta={
                "model_id": self.model_id,
                "instruction_capable": self.instruction_capable,
                "device": str(self.device),
                "raw_output": result["raw_output"],
                "latency_ms": latency_ms,
            },
        )


class Blip2Generator:
    model_id = "blip2"
    instruction_capable = True

    def __init__(self, device=None, adapter_path: Optional[str] = None):
        self.device = device or auto_device()
        self.adapter_path = adapter_path or ""
        self._cache: Dict[tuple[str, str, str], tuple[object, object]] = {}

    def generate(self, image: Image.Image, prompt_config: PromptConfig, params: Dict[str, Any]) -> GenerationResult:
        from fashion_caption.models import blip2

        hf_model_id = params.get("model_id") or "Salesforce/blip2-opt-2.7b"
        adapter_path = params.get("adapter_path") or self.adapter_path
        torch_dtype = params.get("torch_dtype") or ""
        cache_key = (hf_model_id, adapter_path or "", torch_dtype)
        if cache_key not in self._cache:
            self._cache[cache_key] = blip2.load_model(
                self.device,
                model_id=hf_model_id,
                torch_dtype=torch_dtype or None,
                adapter_path=adapter_path or None,
            )

        processor, model = self._cache[cache_key]
        started = time.perf_counter()
        result = blip2.generate_text_details(
            image=image,
            processor=processor,
            model=model,
            model_id=hf_model_id,
            prompt=prompt_config.prompt,
            max_new_tokens=_max_new_tokens(params, 60),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _clean_result(result["description"], prompt_config, params)
        model_id = "blip2-lora" if adapter_path else "blip2"
        return GenerationResult(
            text=text,
            meta={
                "model_id": model_id,
                "hf_model_id": hf_model_id,
                "adapter_path": adapter_path or "",
                "instruction_capable": self.instruction_capable,
                "device": str(self.device),
                "raw_output": result["raw_output"],
                "prompt_id": prompt_config.prompt_id,
                "latency_ms": latency_ms,
            },
        )


class RemoteBlip2Generator:
    instruction_capable = True

    def __init__(self, model_id: str):
        if model_id not in REMOTE_MODEL_IDS:
            raise ValueError(f"Unsupported remote BLIP-2 model: {model_id}")
        self.model_id = model_id

    def generate(self, image: Image.Image, prompt_config: PromptConfig, params: Dict[str, Any]) -> GenerationResult:
        remote_url = params.get("remote_url")
        if not remote_url:
            raise RuntimeError("Remote BLIP-2 URL is not configured.")

        buffer = _image_bytes(image)
        boundary = "----fashion-caption-boundary"
        parts = []

        def add_field(name: str, value: str):
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )

        adapter_path = params.get("adapter_path") if self.model_id == "blip2-lora" else ""
        add_field("model_id", params.get("hf_model_id") or params.get("model_id") or "Salesforce/blip2-opt-2.7b")
        add_field("adapter_path", adapter_path or "")
        add_field("max_new_tokens", str(_max_new_tokens(params, 60)))
        add_field("prompt", prompt_config.prompt)
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="image"; filename="upload.png"\r\n',
                b"Content-Type: image/png\r\n\r\n",
                buffer,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )

        started = time.perf_counter()
        request = urllib.request.Request(
            remote_url,
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=int(params.get("timeout", 300))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote server returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach remote BLIP-2 backend: {exc.reason}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        raw_text = payload.get("description", "")
        text = _clean_result(raw_text, prompt_config, params)
        meta = dict(payload)
        meta.update(
            {
                "model_id": self.model_id,
                "instruction_capable": self.instruction_capable,
                "remote_url": remote_url,
                "latency_ms": latency_ms,
            }
        )
        return GenerationResult(text=text, meta=meta)


def _image_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class GptGenerator:
    model_id = "gpt"
    instruction_capable = True

    def generate(self, image: Image.Image, prompt_config: PromptConfig, params: Dict[str, Any]) -> GenerationResult:
        client = OpenAIResponsesClient(
            model=params.get("openai_model") or params.get("model"),
            timeout=int(params.get("timeout", 90)),
        )
        payload = client.generate_description(
            image=image,
            prompt=prompt_config.prompt,
            max_output_tokens=int(params.get("max_output_tokens", 80)),
            temperature=float(params.get("temperature", 0.2)),
        )
        text = _clean_result(payload["text"], prompt_config, params)
        return GenerationResult(
            text=text,
            meta={
                "model_id": self.model_id,
                "openai_model": payload.get("model"),
                "usage": payload.get("usage") or {},
                "latency_ms": payload.get("latency_ms"),
                "response_id": payload.get("response_id"),
                "api_key_configured": openai_key_configured(),
                "prompt_id": prompt_config.prompt_id,
            },
        )


class ModelRegistry:
    def __init__(self, device=None, prefer_remote_blip2: bool = False):
        self.device = device
        self.prefer_remote_blip2 = prefer_remote_blip2
        self._instances: Dict[str, Any] = {}

    @staticmethod
    def available_model_ids() -> Iterable[str]:
        return tuple(sorted(ALL_MODEL_IDS))

    @staticmethod
    def gpt_configured() -> bool:
        return openai_key_configured()

    def get(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        normalized = model_id.strip().lower()
        if normalized not in ALL_MODEL_IDS:
            raise KeyError(f"Unknown model id: {model_id}")

        use_remote = bool(params.get("remote_url")) and normalized in REMOTE_MODEL_IDS
        cache_key = f"remote:{normalized}" if use_remote or self.prefer_remote_blip2 else normalized
        if cache_key in self._instances:
            return self._instances[cache_key]

        if normalized == "vit-gpt2":
            generator = VitGpt2Generator(device=self.device)
        elif normalized == "blip":
            generator = BlipGenerator(device=self.device)
        elif normalized in {"blip2", "blip2-lora"} and (use_remote or self.prefer_remote_blip2):
            generator = RemoteBlip2Generator(model_id=normalized)
        elif normalized == "blip2":
            generator = Blip2Generator(device=self.device)
        elif normalized == "blip2-lora":
            generator = Blip2Generator(device=self.device, adapter_path=params.get("adapter_path"))
        elif normalized == "gpt":
            generator = GptGenerator()
        else:  # pragma: no cover - normalized guard above keeps this unreachable.
            raise KeyError(f"Unknown model id: {model_id}")

        self._instances[cache_key] = generator
        return generator

    def generate(
        self,
        model_id: str,
        image: Image.Image,
        prompt_config: PromptConfig,
        params: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        params = params or {}
        return self.get(model_id, params=params).generate(image=image, prompt_config=prompt_config, params=params)


__all__ = [
    "Blip2Generator",
    "BlipGenerator",
    "GptGenerator",
    "ModelRegistry",
    "OpenAIConfigurationError",
    "OpenAIGenerationError",
    "RemoteBlip2Generator",
    "VitGpt2Generator",
]
