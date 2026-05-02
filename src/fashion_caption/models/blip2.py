from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from fashion_caption.models.common import load_image
from fashion_caption import config


def load_model(device: torch.device, model_id: str = "Salesforce/blip2-opt-2.7b"):
    """
    Default: blip2-opt-2.7b (public). You can pass blip2-flan-t5-xl/xxl if you have access and enough VRAM.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def caption(
    df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = 60,
    model_id: str = "Salesforce/blip2-flan-t5-base",
) -> pd.DataFrame:
    processor, model = load_model(device, model_id=model_id)
    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLIP-2"):
        image = load_image(row["image_path"])
        inputs = processor(images=image, text=config.PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
        rows.append(
            {
                "id": row["id"],
                "description": text,
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "image_path": str(row["image_path"]),
                "model": "blip2",
                "model_id": model_id,
            }
        )
    return pd.DataFrame(rows)
