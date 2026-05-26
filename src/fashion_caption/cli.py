import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fashion_caption import config
from fashion_caption.preprocessing.data import load_dataset, filter_topwear, split_dataset
from fashion_caption.models import vitgpt2, blip
from fashion_caption.models import blip2
from fashion_caption.visualization.sheets import make_results_sheet
from fashion_caption.eval.metrics import ensure_nltk, bleu1
from fashion_caption.eval.batch import export_generation_csv
from fashion_caption.generation.registry import ModelRegistry
from fashion_caption.postprocess.text import clean_description


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion captioning pipeline (CourseProject-2 rewrite).")
    parser.add_argument("--data-root", default="data", help="Path containing styles.csv and images/ folder.")
    parser.add_argument("--out-dir", default="outputs", help="Where to store CSVs and visualizations.")
    parser.add_argument("--device", default=None, help="Torch device (cpu, cuda, mps). Default: auto-detect.")
    parser.add_argument("--sanity-n", type=int, default=config.SANITY_N, help="Rows for sanity split.")
    parser.add_argument("--eval-n", type=int, default=config.EVAL_N, help="Rows for eval split.")
    parser.add_argument("--skip-vit", action="store_true", help="Skip ViT-GPT2 generation.")
    parser.add_argument("--skip-blip", action="store_true", help="Skip BLIP generation.")
    parser.add_argument("--skip-blip2", action="store_true", help="Skip BLIP-2 generation.")
    parser.add_argument(
        "--blip2-model-id",
        default="Salesforce/blip2-opt-2.7b",
        help="BLIP-2 checkpoint (e.g., Salesforce/blip2-opt-2.7b or blip2-flan-t5-xl if you have access).",
    )
    parser.add_argument("--blip2-adapter-path", default=None, help="Optional path to a LoRA adapter directory.")
    parser.add_argument("--blip2-max-new-tokens", type=int, default=60)
    parser.add_argument("--blip2-torch-dtype", default=None, help="Optional torch dtype for BLIP-2 (e.g., float16).")
    parser.add_argument(
        "--export-generation-csv",
        default=None,
        help="Optional CSV path for batch generation export. When set, runs the export helper and exits.",
    )
    parser.add_argument(
        "--export-models",
        default="vit-gpt2,blip",
        help="Comma-separated model ids for --export-generation-csv (vit-gpt2, blip, blip2, blip2-lora, gpt).",
    )
    parser.add_argument("--export-n", type=int, default=None, help="Optional row count for the batch export.")
    parser.add_argument("--prompt-id", default="ecommerce_v1", help="Prompt template id for generation export.")
    parser.add_argument("--include-metrics", action="store_true", help="Include optional BLEU-1 in export metrics.")
    parser.add_argument("--include-gpt", action="store_true", help="Allow GPT in batch export when explicitly configured.")
    parser.add_argument("--remote-url", default=None, help="Optional remote BLIP-2 /caption endpoint for batch export.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading dataset...")
    df = load_dataset(data_root)
    df_topwear = filter_topwear(df)
    df_sanity, df_eval = split_dataset(df_topwear, sanity_n=args.sanity_n, eval_n=args.eval_n, seed=config.SEED)
    print(f"Sanity set: {len(df_sanity)} rows; eval set: {len(df_eval)} rows.")

    if args.export_generation_csv:
        export_df = df_sanity.iloc[: args.export_n].copy() if args.export_n else df_sanity
        export_models = [item.strip() for item in args.export_models.split(",") if item.strip()]
        export_models = [
            model for model in export_models if model != "gpt" or (args.include_gpt and ModelRegistry.gpt_configured())
        ]
        metrics = export_generation_csv(
            df=export_df,
            out_csv=Path(args.export_generation_csv).resolve(),
            model_ids=export_models,
            prompt_id=args.prompt_id,
            include_metrics=args.include_metrics,
            device=device,
            params={
                "remote_url": args.remote_url,
                "hf_model_id": args.blip2_model_id,
                "adapter_path": args.blip2_adapter_path,
                "torch_dtype": args.blip2_torch_dtype,
                "max_new_tokens": args.blip2_max_new_tokens,
            },
        )
        print(f"Saved generation export to {Path(args.export_generation_csv).resolve()}")
        print(json.dumps(metrics, indent=2))
        return

    vit_results = None
    blip_results = None
    blip2_results = None

    if not args.skip_vit:
        vit_results = vitgpt2.caption(df_sanity, device=device)
        vit_results["description_raw"] = vit_results["description"]
        vit_results["description"] = vit_results.apply(
            lambda r: clean_description(
                r["description_raw"],
                prompt=config.PROMPT,
                article_type=r.get("articleType"),
                base_colour=r.get("baseColour"),
            ),
            axis=1,
        )
        vit_results.to_csv(out_dir / "df_vitgpt2_sanity.csv", index=False)

    if not args.skip_blip:
        blip_results = blip.caption(df_sanity, device=device)
        blip_results["description_raw"] = blip_results["description"]
        blip_results["description"] = blip_results.apply(
            lambda r: clean_description(
                r["description_raw"],
                prompt=config.PROMPT,
                article_type=r.get("articleType"),
                base_colour=r.get("baseColour"),
            ),
            axis=1,
        )
        blip_results.to_csv(out_dir / "df_blip_sanity.csv", index=False)

    if not args.skip_blip2:
        blip2_results = blip2.caption(
            df_sanity,
            device=device,
            model_id=args.blip2_model_id,
            max_new_tokens=args.blip2_max_new_tokens,
            torch_dtype=args.blip2_torch_dtype,
            adapter_path=args.blip2_adapter_path,
        )
        blip2_results["description_raw"] = blip2_results["description"]
        blip2_results["description"] = blip2_results.apply(
            lambda r: clean_description(
                r["description_raw"],
                prompt=config.PROMPT,
                article_type=r.get("articleType"),
                base_colour=r.get("baseColour"),
            ),
            axis=1,
        )
        blip2_results.to_csv(out_dir / "df_blip2_sanity.csv", index=False)

    if vit_results is not None and blip_results is not None:
        df_compare = vit_results.merge(
            blip_results[["id", "description"]],
            on="id",
            suffixes=("_vit", "_blip"),
        )
        df_compare.to_csv(out_dir / "df_compare_sanity.csv", index=False)
        sheet_path = make_results_sheet(df_compare, out_dir / "experimental_results.png", n=8, cols=2)
        print(f"Saved comparison sheet to {sheet_path}")

    ensure_nltk()
    if not args.skip_vit:
        df_eval_vit = vitgpt2.caption(df_eval, device=device)
        df_eval_vit = df_eval[["id", "productDisplayName"]].merge(df_eval_vit, on="id", how="inner")
        df_eval_vit["reference"] = df_eval_vit["productDisplayName"]
        df_eval_vit["generated"] = df_eval_vit["description"]
        bleu_vit = bleu1(df_eval_vit)
        df_eval_vit.to_csv(out_dir / "df_vitgpt2_eval.csv", index=False)
        print(f"BLEU-1 (ViT-GPT2, eval split): {bleu_vit:.4f}")

    if not args.skip_blip:
        df_eval_blip = blip.caption(df_eval, device=device)
        df_eval_blip = df_eval[["id", "productDisplayName"]].merge(df_eval_blip, on="id", how="inner")
        df_eval_blip["reference"] = df_eval_blip["productDisplayName"]
        df_eval_blip["generated"] = df_eval_blip["description"]
        bleu_blip = bleu1(df_eval_blip)
        df_eval_blip.to_csv(out_dir / "df_blip_eval.csv", index=False)
        print(f"BLEU-1 (BLIP, eval split): {bleu_blip:.4f}")

    if not args.skip_blip2:
        df_eval_blip2 = blip2.caption(
            df_eval,
            device=device,
            model_id=args.blip2_model_id,
            max_new_tokens=args.blip2_max_new_tokens,
            torch_dtype=args.blip2_torch_dtype,
            adapter_path=args.blip2_adapter_path,
        )
        df_eval_blip2 = df_eval[["id", "productDisplayName"]].merge(df_eval_blip2, on="id", how="inner")
        df_eval_blip2["reference"] = df_eval_blip2["productDisplayName"]
        df_eval_blip2["generated"] = df_eval_blip2["description"]
        bleu_blip2 = bleu1(df_eval_blip2)
        df_eval_blip2.to_csv(out_dir / "df_blip2_eval.csv", index=False)
        print(f"BLEU-1 (BLIP-2, eval split): {bleu_blip2:.4f}")


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
