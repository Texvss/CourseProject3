from pathlib import Path
from typing import Tuple

import pandas as pd

from fashion_caption import config


def load_dataset(data_root: Path) -> pd.DataFrame:
    styles_path = data_root / "styles.csv"
    images_path = data_root / "images"
    if not styles_path.exists():
        raise FileNotFoundError(f"Missing styles.csv at {styles_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Missing images directory at {images_path}")

    df = pd.read_csv(styles_path, on_bad_lines="skip")
    df["id"] = df["id"].astype(str)
    df["image_path"] = df["id"].apply(lambda x: images_path / f"{x}.jpg")
    df["has_image"] = df["image_path"].apply(lambda p: p.exists())

    missing = (~df["has_image"]).sum()
    if missing:
        print(f"Skipping {missing} rows without images.")
    df = df[df["has_image"]].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def filter_topwear(df: pd.DataFrame) -> pd.DataFrame:
    df_topwear = df[df["subCategory"] == "Topwear"]
    df_topwear = df_topwear[df_topwear["articleType"].isin(config.TOPWEAR_TYPES)]
    df_topwear = df_topwear.copy().reset_index(drop=True)
    return df_topwear


def split_dataset(
    df: pd.DataFrame, sanity_n: int = config.SANITY_N, eval_n: int = config.EVAL_N, seed: int = config.SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_sanity = df_shuffled.iloc[:sanity_n].copy()
    df_eval = df_shuffled.iloc[sanity_n : sanity_n + eval_n].copy()
    return df_sanity, df_eval
