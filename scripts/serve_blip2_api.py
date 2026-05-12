"""
Small FastAPI server for BLIP-2 / BLIP-2 LoRA inference on RunPod.

Example:
    PYTHONPATH=src python scripts/serve_blip2_api.py \
        --host 0.0.0.0 \
        --port 8000 \
        --device cuda \
        --model-id Salesforce/blip2-opt-2.7b
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption.models import blip2  # noqa: E402

DEFAULT_API_PROMPT = (
    "Describe the garment for an e-commerce catalog in one short sentence. "
    "Mention the clothing type, main color, and visible details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BLIP-2 inference API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Torch device. Default: cuda if available, else cpu.")
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--adapter-path", default=None, help="Optional default LoRA adapter path.")
    parser.add_argument("--torch-dtype", default=None, help="Optional torch dtype, e.g. float16.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--prompt", default=DEFAULT_API_PROMPT)
    return parser.parse_args()


ARGS = parse_args()
DEVICE = torch.device(ARGS.device if ARGS.device else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[object, object]] = {}

app = FastAPI(title="Fashion Captioning BLIP-2 API")


def get_model_bundle(model_id: str, adapter_path: Optional[str], torch_dtype: Optional[str]):
    cache_key = (model_id, adapter_path or "", torch_dtype or "")
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = blip2.load_model(
            DEVICE,
            model_id=model_id,
            torch_dtype=torch_dtype,
            adapter_path=adapter_path,
        )
    return MODEL_CACHE[cache_key]


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "device": str(DEVICE),
        "default_model_id": ARGS.model_id,
        "default_adapter_path": ARGS.adapter_path or "",
        "default_prompt": ARGS.prompt,
        "cache_size": len(MODEL_CACHE),
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "fashion-captioning-blip2-api",
        "ok": True,
        "caption_endpoint": "/caption",
        "health_endpoint": "/health",
    }


@app.post("/caption")
async def caption(
    image: UploadFile = File(...),
    model_id: Optional[str] = Form(default=None),
    adapter_path: Optional[str] = Form(default=None),
    torch_dtype: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    max_new_tokens: Optional[int] = Form(default=None),
) -> dict:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    resolved_model_id = model_id or ARGS.model_id
    resolved_adapter_path = adapter_path if adapter_path is not None else ARGS.adapter_path
    resolved_torch_dtype = torch_dtype if torch_dtype is not None else ARGS.torch_dtype
    resolved_max_new_tokens = max_new_tokens or ARGS.max_new_tokens

    if resolved_adapter_path:
        adapter_dir = Path(resolved_adapter_path)
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise HTTPException(
                status_code=400,
                detail=f"LoRA adapter config not found at {adapter_config}",
            )

    try:
        processor, model = get_model_bundle(
            model_id=resolved_model_id,
            adapter_path=resolved_adapter_path,
            torch_dtype=resolved_torch_dtype,
        )
        result = blip2.generate_text_details(
            image=pil_image,
            processor=processor,
            model=model,
            model_id=resolved_model_id,
            prompt=prompt or ARGS.prompt,
            max_new_tokens=resolved_max_new_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive path for remote debugging
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "model_id": resolved_model_id,
                "adapter_path": resolved_adapter_path or "",
            },
        )

    return {
        "description": result["description"],
        "raw_description": result["raw_description"],
        "device": str(DEVICE),
        "model": "blip2",
        "model_id": resolved_model_id,
        "adapter_path": resolved_adapter_path or "",
        "prompt": result["prompt"],
        "max_new_tokens": resolved_max_new_tokens,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
