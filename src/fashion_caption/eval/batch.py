from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from fashion_caption.eval.metrics import attribute_accuracy, attribute_metrics, bleu1, protocol_metrics
from fashion_caption.generation import ModelRegistry
from fashion_caption.models.common import load_image
from fashion_caption.prompts import get_prompt_config


EXPORT_COLUMNS = [
    "id",
    "image_path",
    "articleType",
    "baseColour",
    "productDisplayName",
    "model_id",
    "prompt_id",
    "prompt",
    "raw_output",
    "cleaned_output",
    "generated_text",
    "error",
]


def compact_error(exc: Exception | str) -> str:
    return " ".join(str(exc).split())


def export_generation_csv(
    df: pd.DataFrame,
    out_csv: Path,
    model_ids: Iterable[str],
    prompt_id: str = "ecommerce_v1",
    params: Optional[Dict[str, object]] = None,
    include_metrics: bool = False,
    device=None,
    fail_fast: bool = True,
    skip_failed_models: bool = True,
) -> dict:
    params = params or {}
    registry = ModelRegistry(device=device, prefer_remote_blip2=bool(params.get("remote_url")))
    rows = []
    failed_models: Dict[str, str] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Batch generation"):
        image = load_image(Path(row["image_path"]))
        for model_id in model_ids:
            prompt_config = get_prompt_config(
                prompt_id=prompt_id,
                article_type=row.get("articleType", ""),
                base_colour=row.get("baseColour", ""),
            )
            record = {
                "id": row.get("id", ""),
                "image_path": str(row.get("image_path", "")),
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "productDisplayName": row.get("productDisplayName", ""),
                "model_id": model_id,
                "prompt_id": prompt_id,
                "prompt": prompt_config.prompt,
                "raw_output": "",
                "cleaned_output": "",
                "generated_text": "",
                "error": "",
            }
            if model_id in failed_models and skip_failed_models:
                record["error"] = f"skipped after previous failure: {failed_models[model_id]}"
                rows.append(record)
                pd.DataFrame(rows, columns=EXPORT_COLUMNS).to_csv(out_csv, index=False)
                continue
            try:
                result = registry.generate(
                    model_id=model_id,
                    image=image,
                    prompt_config=prompt_config,
                    params=params,
                )
                raw_output = result.meta.get("raw_description") or result.meta.get("raw_output") or result.text
                record["raw_output"] = raw_output
                record["cleaned_output"] = result.text
                record["generated_text"] = result.text
            except Exception as exc:
                record["error"] = compact_error(exc)
                failed_models[model_id] = record["error"]
                rows.append(record)
                pd.DataFrame(rows, columns=EXPORT_COLUMNS).to_csv(out_csv, index=False)
                if fail_fast:
                    raise
                continue
            rows.append(record)
            pd.DataFrame(rows, columns=EXPORT_COLUMNS).to_csv(out_csv, index=False)

    out_df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    out_df.to_csv(out_csv, index=False)

    metrics = attribute_accuracy(out_df, gen_col="cleaned_output")
    if include_metrics and "productDisplayName" in df.columns:
        scored = out_df.rename(columns={"productDisplayName": "reference"})
        scored["generated"] = scored["cleaned_output"]
        metrics["bleu1"] = bleu1(scored, ref_col="reference", gen_col="generated")
    return metrics


def _blank_attribute_metrics(attrs: dict) -> dict:
    blanked = {}
    for key in attrs:
        blanked[key] = np.nan
    return blanked


def compute_generation_metrics(df: pd.DataFrame, split: str = "") -> pd.DataFrame:
    rows = []
    if "error" in df.columns:
        valid = df[df["error"].fillna("").astype(str).eq("")].copy()
    else:
        valid = df.copy()
    for model_id, model_df in valid.groupby("model_id", sort=False):
        for variant, gen_col in (("raw", "raw_output"), ("cleaned", "cleaned_output")):
            scored = model_df.copy()
            scored["reference"] = scored.get("productDisplayName", "")
            scored["generated"] = scored.get(gen_col, "")
            attrs = attribute_metrics(scored, gen_col=gen_col)
            keyword_note = ""
            if variant == "cleaned":
                attrs = _blank_attribute_metrics(attrs)
                keyword_note = "omitted_by_construction"
            rows.append(
                {
                    "split": split,
                    "model_id": model_id,
                    "text_variant": variant,
                    "rows": int(len(scored)),
                    "bleu1": bleu1(scored, ref_col="reference", gen_col="generated"),
                    "keyword_metrics_note": keyword_note,
                    **attrs,
                    **protocol_metrics(scored, gen_col=gen_col, prompt_col="prompt"),
                }
            )
    return pd.DataFrame(rows)
