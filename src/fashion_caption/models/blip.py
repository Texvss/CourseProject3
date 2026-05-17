from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from fashion_caption.models.common import load_image


def load_model(device: torch.device):
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def generate_text_details(
    image: Image.Image,
    processor,
    model,
    max_new_tokens: int = 40,
) -> Dict[str, str]:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return {
        "description": text,
        "raw_description": text,
        "raw_output": text,
    }


def generate_text(
    image: Image.Image,
    processor,
    model,
    max_new_tokens: int = 40,
) -> str:
    return generate_text_details(
        image=image,
        processor=processor,
        model=model,
        max_new_tokens=max_new_tokens,
    )["description"]


def caption(
    df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = 40,
) -> pd.DataFrame:
    processor, model = load_model(device)
    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLIP"):
        image = load_image(row["image_path"])
        text = generate_text(
            image=image,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
        )
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
