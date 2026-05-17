from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from tqdm import tqdm

from fashion_caption.eval.metrics import attribute_accuracy, bleu1
from fashion_caption.generation import ModelRegistry
from fashion_caption.models.common import load_image
from fashion_caption.prompts import get_prompt_config


EXPORT_COLUMNS = [
    "id",
    "image_path",
    "articleType",
    "baseColour",
    "model_id",
    "prompt_id",
    "generated_text",
]


def export_generation_csv(
    df: pd.DataFrame,
    out_csv: Path,
    model_ids: Iterable[str],
    prompt_id: str = "ecommerce_v1",
    params: Optional[Dict[str, object]] = None,
    include_metrics: bool = False,
    device=None,
) -> dict:
    params = params or {}
    registry = ModelRegistry(device=device, prefer_remote_blip2=bool(params.get("remote_url")))
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Batch generation"):
        image = load_image(Path(row["image_path"]))
        for model_id in model_ids:
            prompt_config = get_prompt_config(
                prompt_id=prompt_id,
                article_type=row.get("articleType", ""),
                base_colour=row.get("baseColour", ""),
            )
            result = registry.generate(
                model_id=model_id,
                image=image,
                prompt_config=prompt_config,
                params=params,
            )
            rows.append(
                {
                    "id": row.get("id", ""),
                    "image_path": str(row.get("image_path", "")),
                    "articleType": row.get("articleType", ""),
                    "baseColour": row.get("baseColour", ""),
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "generated_text": result.text,
                }
            )

    out_df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    metrics = attribute_accuracy(out_df)
    if include_metrics and "productDisplayName" in df.columns:
        refs = df[["id", "productDisplayName"]].rename(columns={"productDisplayName": "reference"})
        scored = out_df.merge(refs, on="id", how="left")
        scored["generated"] = scored["generated_text"]
        metrics["bleu1"] = bleu1(scored, ref_col="reference", gen_col="generated")
    return metrics
