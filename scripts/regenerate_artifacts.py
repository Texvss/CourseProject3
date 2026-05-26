from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption import config
from fashion_caption.eval.batch import compute_generation_metrics, export_generation_csv
from fashion_caption.generation.registry import ModelRegistry
from fashion_caption.preprocessing.data import filter_topwear, load_dataset, split_dataset


MODEL_FILE_STEMS = {
    "vit-gpt2": "vitgpt2",
    "blip": "blip",
    "blip2": "blip2",
    "blip2-lora": "blip2-lora",
    "gpt": "gpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate report CSVs, metrics, and plots.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sanity-n", type=int, default=config.SANITY_N)
    parser.add_argument("--eval-n", type=int, default=config.EVAL_N)
    parser.add_argument("--models", default="vit-gpt2,blip")
    parser.add_argument("--prompt-id", default="ecommerce_v1")
    parser.add_argument("--remote-url", default=os.environ.get("FASHION_CAPTION_REMOTE_URL"))
    parser.add_argument("--blip2-model-id", default=os.environ.get("FASHION_CAPTION_REMOTE_MODEL_ID", "Salesforce/blip2-opt-2.7b"))
    parser.add_argument("--blip2-adapter-path", default=os.environ.get("FASHION_CAPTION_REMOTE_ADAPTER_PATH"))
    parser.add_argument("--blip2-torch-dtype", default=os.environ.get("FASHION_CAPTION_TORCH_DTYPE"))
    parser.add_argument("--blip2-quant", default=os.environ.get("BLIP2_QUANT", "none"))
    parser.add_argument("--blip2-max-new-tokens", type=int, default=60)
    parser.add_argument("--include-gpt", action="store_true", help="Evaluate GPT only if it is explicitly configured.")
    parser.add_argument("--reuse-existing", action="store_true", help="Rebuild metrics and plots from existing df_all CSVs.")
    return parser.parse_args()


def resolve_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def selected_models(raw: str, include_gpt: bool) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if include_gpt and ModelRegistry.gpt_configured() and "gpt" not in models:
        models.append("gpt")
    return [model for model in models if model != "gpt" or (include_gpt and ModelRegistry.gpt_configured())]


def write_per_model_csvs(df: pd.DataFrame, split: str, out_dir: Path) -> None:
    for model_id, model_df in df.groupby("model_id", sort=False):
        stem = MODEL_FILE_STEMS.get(model_id, model_id.replace("-", ""))
        model_df.to_csv(out_dir / f"df_{stem}_{split}.csv", index=False)


def write_metrics_plot(metrics: pd.DataFrame, split: str, out_dir: Path) -> None:
    if metrics.empty or "text_variant" not in metrics.columns:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cleaned = metrics[metrics["text_variant"].eq("cleaned")].copy()
    if cleaned.empty:
        return
    plot_df = cleaned.set_index("model_id")[
        ["bleu1", "protocol_compliance_rate", "forbidden_mentions_rate", "prompt_echo_rate"]
    ]
    ax = plot_df.plot(kind="bar", figsize=(10.5, 5.2), ylim=(0, 1.12), rot=0)
    ax.set_title(f"{split.capitalize()} cleaned-output service metrics", fontsize=11, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(
        ["BLEU-1", "Protocol compliant", "Forbidden mention", "Prompt echo"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_dir / f"{split}_metrics.png", dpi=180)
    plt.close()


def summarize_availability(df: pd.DataFrame, models: Iterable[str]) -> dict:
    summary = {}
    for model_id in models:
        model_df = df[df["model_id"].eq(model_id)]
        errors = model_df["error"].fillna("").astype(str)
        ok_rows = int(errors.eq("").sum())
        first_error = next((err for err in errors if err), "")
        summary[model_id] = {
            "rows": int(len(model_df)),
            "ok_rows": ok_rows,
            "available": ok_rows > 0,
            "first_error": first_error,
        }
    return summary


def run() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    models = selected_models(args.models, include_gpt=args.include_gpt)
    df = filter_topwear(load_dataset(data_root))
    sanity_df, eval_df = split_dataset(df, sanity_n=args.sanity_n, eval_n=args.eval_n, seed=config.SEED)
    params = {
        "remote_url": args.remote_url,
        "hf_model_id": args.blip2_model_id,
        "adapter_path": args.blip2_adapter_path,
        "torch_dtype": args.blip2_torch_dtype,
        "quant": args.blip2_quant,
        "max_new_tokens": args.blip2_max_new_tokens,
    }

    status = {
        "device": str(device),
        "sanity_rows": int(len(sanity_df)),
        "eval_rows": int(len(eval_df)),
        "requested_models": models,
        "gpt_evaluated": "gpt" in models,
        "gpt_note": "GPT excluded unless --include-gpt is passed and ENABLE_GPT_API=1 plus OPENAI_API_KEY are set.",
    }

    for split, split_df in (("sanity", sanity_df), ("eval", eval_df)):
        combined_path = out_dir / f"df_all_{split}.csv"
        if not args.reuse_existing:
            export_generation_csv(
                df=split_df,
                out_csv=combined_path,
                model_ids=models,
                prompt_id=args.prompt_id,
                params=params,
                include_metrics=False,
                device=device,
                fail_fast=False,
            )
        elif not combined_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing artifact: {combined_path}")
        generated = pd.read_csv(combined_path)
        if "error" in generated.columns:
            generated["error"] = generated["error"].fillna("").astype(str).map(lambda value: " ".join(value.split()))
            generated.to_csv(combined_path, index=False)
        write_per_model_csvs(generated, split, out_dir)
        metrics = compute_generation_metrics(generated, split=split)
        metrics.to_csv(out_dir / f"metrics_{split}.csv", index=False)
        write_metrics_plot(metrics, split, out_dir)
        status[f"{split}_availability"] = summarize_availability(generated, models)

    status_path = out_dir / "artifact_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    run()
