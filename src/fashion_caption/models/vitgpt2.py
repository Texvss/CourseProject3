from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from fashion_caption.models.common import load_image


def load_model(device: torch.device):
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, tokenizer, model


def caption(
    df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = 40,
    num_beams: int = 3,
) -> pd.DataFrame:
    processor, tokenizer, model = load_model(device)
    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ViT-GPT2"):
        image = load_image(row["image_path"])
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        rows.append(
            {
                "id": row["id"],
                "description": text,
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "image_path": str(row["image_path"]),
                "model": "vit-gpt2",
            }
        )
    return pd.DataFrame(rows)
