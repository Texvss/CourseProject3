from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:
    raise SystemExit("Install peft and accelerate to run LoRA fine-tuning.") from exc


def get_version(package_name: str) -> str | None:
    try:
        return md.version(package_name)
    except md.PackageNotFoundError:
        return None


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def remove_incompatible_torchao() -> None:
    torchao_version = get_version("torchao")
    if torchao_version is None:
        return
    if version_tuple(torchao_version) >= (0, 16, 0):
        return
    print(f"Removing incompatible torchao=={torchao_version}")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_article_type(value: str) -> str:
    text = normalize_spaces(value)
    if not text:
        return "item"
    return text.replace("Tshirts", "t-shirt").replace("Shirts", "shirt").replace("Tops", "top")


def template_target_description(row: pd.Series) -> str:
    name = normalize_spaces(row.get("productDisplayName", ""))
    if not name:
        return "A catalog-style garment with visible details."
    return name


def sanitize_target_description(text: str, row: pd.Series) -> str:
    cleaned = normalize_spaces(text)
    cleaned = re.sub(
        r"\b(model|posing|wearing|standing|showing|pictured|photo|image|background|"
        r"men|mens|women|womens|boys|girls|unisex|brand|price|sale|discount|cheap|premium)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = normalize_spaces(cleaned)
    cleaned = cleaned.strip(" -,:;")
    if cleaned and not cleaned.lower().startswith(("a ", "an ")):
        cleaned = f"A {cleaned[0].lower()}{cleaned[1:]}."
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned = f"{cleaned}."
    return normalize_spaces(cleaned)


def make_training_prompt(row: pd.Series) -> str:
    return (
        "Write a short e-commerce garment description in one sentence. "
        "Mention only visible clothing details. Do not mention people, poses, or background."
    )


def resolve_dtype(device: torch.device, model_id: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if "flan-t5" in model_id.lower() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_task_type(model_id: str) -> TaskType:
    if "flan-t5" in model_id.lower():
        return TaskType.SEQ_2_SEQ_LM
    return TaskType.CAUSAL_LM


def resolve_target_modules(model_id: str) -> list[str]:
    if "opt" in model_id.lower():
        return ["q_proj", "v_proj"]
    return ["q", "v"]


def build_train_df(data_root: Path, max_samples: int, seed: int) -> pd.DataFrame:
    styles_path = data_root / "styles.csv"
    images_root = data_root / "images"
    df = pd.read_csv(styles_path, on_bad_lines="skip")
    df["id"] = df["id"].astype(str)
    df["image_path"] = df["id"].apply(lambda x: images_root / f"{x}.jpg")
    df = df[df["image_path"].apply(lambda p: p.exists())].copy()

    if "subCategory" in df.columns:
        df = df[df["subCategory"] == "Topwear"].copy()

    df["target_description"] = df.apply(template_target_description, axis=1)
    df["target_description"] = df.apply(
        lambda row: sanitize_target_description(row["target_description"], row),
        axis=1,
    )
    df = df[df["target_description"].astype(str).str.len().gt(15)].copy()
    return df.sample(frac=1.0, random_state=seed).head(max_samples).reset_index(drop=True)


class FashionCaptionLoRADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: Blip2Processor,
        max_prompt_length: int,
        max_label_length: int,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(str(row["image_path"])).convert("RGB")
        prompt = make_training_prompt(row)
        target = normalize_spaces(str(row["target_description"]))

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return {key: value.squeeze(0) for key, value in inputs.items()}


def train(args: argparse.Namespace) -> None:
    remove_incompatible_torchao()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_df = build_train_df(Path(args.data_root), args.max_samples, args.seed)
    print(f"Training rows: {len(train_df)}")

    dtype = resolve_dtype(device, args.model_id)
    processor = Blip2Processor.from_pretrained(args.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=resolve_target_modules(args.model_id),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=resolve_task_type(args.model_id),
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    dataset = FashionCaptionLoRADataset(
        train_df,
        processor,
        max_prompt_length=args.max_prompt_length,
        max_label_length=args.max_label_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, (len(loader) * args.epochs) // max(1, args.grad_accum_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"LoRA epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {
                key: (
                    value.to(device=device, dtype=dtype)
                    if torch.is_floating_point(value)
                    else value.to(device)
                )
                for key, value in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss / max(1, args.grad_accum_steps)
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({"loss": float(loss.detach().cpu()) * max(1, args.grad_accum_steps)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Saved LoRA adapter to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for BLIP-2 on the fashion dataset.")
    parser.add_argument("--data-root", default="data", help="Folder with styles.csv and images/")
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b", help="Base BLIP-2 checkpoint.")
    parser.add_argument("--out-dir", default="lora-blip2-ecommerce", help="Where to save LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=96, help="Limit samples for quick runs.")
    parser.add_argument("--max-prompt-length", type=int, default=48)
    parser.add_argument("--max-label-length", type=int, default=48)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
