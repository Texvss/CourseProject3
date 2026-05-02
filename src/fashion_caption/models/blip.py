from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from fashion_caption.models.common import load_image


def load_model(device: torch.device):
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def caption(
    df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = 40,
) -> pd.DataFrame:
    processor, model = load_model(device)
    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLIP"):
        image = load_image(row["image_path"])
        inputs = processor(images=image, return_tensors="pt").to(device)
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
                "model": "blip-image-captioning-base",
            }
        )
    return pd.DataFrame(rows)
