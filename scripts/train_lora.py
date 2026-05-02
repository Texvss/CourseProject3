"""
Minimal LoRA fine-tuning script for BLIP-2 on your dataset.

Assumptions:
- Dataset: data/styles.csv with columns id, productDisplayName (reference text), image_path (filled via preprocessing).
- Uses LoRA on the text decoder of blip2-flan-t5-base to keep memory small.

Run (example on GPU):
    HF_HOME=.cache/huggingface \
    accelerate launch scripts/train_lora.py \
        --data-root data \
        --epochs 1 \
        --batch-size 2 \
        --lr 5e-5 \
        --out-dir lora-blip2

Notes:
- On Apple Silicon / CPU training will be slow; prefer a GPU runtime (Colab/Gradient).
- bitsandbytes is optional and may be unavailable on macOS; script falls back to full precision.
"""
import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    raise SystemExit("Install peft to run LoRA fine-tuning: pip install peft")

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


class CaptionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(
            images=image,
            text=str(row["reference"]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        # flatten batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    load_kwargs = {}
    if BNB_AVAILABLE and device.type == "cuda":
        load_kwargs = {"load_in_8bit": True, "device_map": "auto"}
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **load_kwargs)

    if BNB_AVAILABLE and device.type == "cuda":
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v", "k"],  # broad but safe for T5 blocks
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # Data
    styles_path = Path(args.data_root) / "styles.csv"
    df = pd.read_csv(styles_path, on_bad_lines="skip")
    df["id"] = df["id"].astype(str)
    df["image_path"] = df["id"].apply(lambda x: Path(args.data_root) / "images" / f"{x}.jpg")
    df = df[df["image_path"].apply(lambda p: p.exists())]
    df["reference"] = df["productDisplayName"].fillna("")
    df = df.sample(frac=1.0, random_state=42).head(args.max_samples)

    dataset = CaptionDataset(df, processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(10, total_steps // 10), num_training_steps=total_steps)

    model.to(device)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Saved LoRA-adapted model to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for BLIP-2.")
    parser.add_argument("--data-root", default="data", help="Folder with styles.csv and images/")
    parser.add_argument("--model-id", default="Salesforce/blip2-flan-t5-base", help="Base BLIP-2 checkpoint.")
    parser.add_argument("--out-dir", default="lora-blip2", help="Where to save adapters.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=2000, help="Limit samples for quick runs.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
