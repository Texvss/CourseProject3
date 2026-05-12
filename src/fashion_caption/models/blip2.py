from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
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


def _strip_prompt_echo(text: str, prompt: str) -> str:
    text = text.strip()
    prompt = prompt.strip()
    if prompt and text.lower().startswith(prompt.lower()):
        return text[len(prompt) :].strip(" \n\t:.-")
    return text


def generate_text_details(
    image: Image.Image,
    processor,
    model,
    model_id: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 60,
) -> Dict[str, str]:
    prompt_text = prompt or config.PROMPT
    inputs = processor(images=image.convert("RGB"), text=prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw_output = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    raw_description = _strip_prompt_echo(raw_output, prompt_text)
    return {
        "description": raw_description,
        "raw_description": raw_description,
        "raw_output": raw_output,
        "prompt": prompt_text,
    }


def generate_text(
    image: Image.Image,
    processor,
    model,
    model_id: str,
    prompt: Optional[str] = None,
    max_new_tokens: int = 60,
) -> str:
    return generate_text_details(
        image=image,
        processor=processor,
        model=model,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )["description"]


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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLIP-2"):
        image = load_image(row["image_path"])
        text = generate_text(
            image=image,
            processor=processor,
            model=model,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
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
