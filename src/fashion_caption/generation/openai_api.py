from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from PIL import Image


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class OpenAIConfigurationError(RuntimeError):
    pass


class OpenAIGenerationError(RuntimeError):
    pass


def openai_key_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def gpt_feature_enabled() -> bool:
    return os.environ.get("ENABLE_GPT_API", "").strip() == "1"


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 90,
    ):
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("OPENAI_GPT_MODEL") or DEFAULT_OPENAI_MODEL).strip()
        self.timeout = timeout

    def generate_description(
        self,
        image: Image.Image,
        prompt: str,
        max_output_tokens: int = 80,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not gpt_feature_enabled():
            raise OpenAIConfigurationError("GPT generation is disabled. Set ENABLE_GPT_API=1 to enable it.")
        if not self.api_key:
            raise OpenAIConfigurationError(
                "OPENAI_API_KEY is not configured. Add it to your environment before using GPT generation."
            )

        request_body = {
            "model": self.model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Generate the product description from this image and return only the description.",
                        },
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(image),
                            "detail": "auto",
                        },
                    ],
                },
            ],
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        encoded = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            try:
                parsed = json.loads(detail)
                message = parsed.get("error", {}).get("message") or detail
            except json.JSONDecodeError:
                message = detail or str(exc)
            raise OpenAIGenerationError(f"OpenAI API returned {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise OpenAIGenerationError(f"Failed to reach OpenAI API: {exc.reason}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "text": extract_response_text(payload),
            "usage": payload.get("usage") or {},
            "latency_ms": latency_ms,
            "model": payload.get("model") or self.model,
            "response_id": payload.get("id"),
        }
