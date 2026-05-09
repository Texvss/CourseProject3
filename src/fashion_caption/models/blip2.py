import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from fashion_caption.models.common import load_image
from fashion_caption import config


def _resolve_torch_dtype(torch_dtype: Optional[str], device: torch.device, model_id: str):
    if torch_dtype:
        if not hasattr(torch, torch_dtype):
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")
        return getattr(torch, torch_dtype)
    if device.type == "cuda":
        if "flan-t5" in model_id.lower() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _decode_generated_text(processor, inputs, output_ids: torch.Tensor, model_id: str) -> str:
    # BLIP-2 OPT returns prompt + generated continuation. We only want the generated suffix.
    if "opt" in model_id.lower() and "input_ids" in inputs:
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        if generated_ids.shape[1] > 0:
            return processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def _clean_generated_description(text: str) -> str:
    cleaned = _normalize_spaces(text)
    cleaned = cleaned.replace("Product description:", "").strip()
    cleaned = re.sub(r"^(write|describe)\b.*?:", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"\b(model|posing|wearing|standing|showing|pictured|photo|image|background|photography)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return _normalize_spaces(cleaned)


def load_model(
    device: torch.device,
    model_id: str = "Salesforce/blip2-opt-2.7b",
    torch_dtype: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """
    Default: blip2-opt-2.7b (public). You can pass blip2-flan-t5-xl/xxl if you have access and enough VRAM.
    """
    dtype = _resolve_torch_dtype(torch_dtype, device, model_id)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, **model_kwargs).to(device)
    if adapter_path:
        adapter_dir = Path(adapter_path)
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(f"LoRA adapter config not found at {adapter_config}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft to load a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir)).to(device)
    model.eval()
    return processor, model


def caption(
    df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = 60,
    model_id: str = "Salesforce/blip2-opt-2.7b",
    torch_dtype: Optional[str] = None,
    prompt: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> pd.DataFrame:
    processor, model = load_model(
        device,
        model_id=model_id,
        torch_dtype=torch_dtype,
        adapter_path=adapter_path,
    )
    rows: List[dict] = []
    prompt_text = prompt or config.PROMPT
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLIP-2"):
        image = load_image(row["image_path"])
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = _decode_generated_text(processor, inputs, out_ids, model_id=model_id)
        text = _clean_generated_description(text)
        rows.append(
            {
                "id": row["id"],
                "description": text,
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "image_path": str(row["image_path"]),
                "model": "blip2",
                "model_id": model_id,
                "adapter_path": adapter_path or "",
            }
        )
    return pd.DataFrame(rows)
