import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import pandas as pd


def make_results_sheet(
    df: pd.DataFrame,
    out_path: Path,
    n: int = 8,
    cols: int = 2,
    img_size: Tuple[int, int] = (256, 256),
    wrap_width: int = 42,
) -> Path:
    """
    Save a side-by-side comparison sheet for ViT-GPT2 and BLIP outputs.
    """
    df = df.head(n).copy()
    rows = (len(df) + cols - 1) // cols
    fig = plt.figure(figsize=(12, rows * 4.2), dpi=200)
    fig.patch.set_facecolor("white")

    for idx, (_, r) in enumerate(df.iterrows()):
        rr = idx // cols
        cc = idx % cols
        block_left = cc / cols
        block_right = (cc + 1) / cols
        block_top = 1 - rr / rows
        block_bottom = 1 - (rr + 1) / rows
        pad_x, pad_y = 0.02, 0.05

        ax_img = fig.add_axes(
            [
                block_left + pad_x,
                block_bottom + pad_y,
                (block_right - block_left) * 0.38,
                (block_top - block_bottom) * 0.85,
            ]
        )
        ax_img.axis("off")

        ax_txt = fig.add_axes(
            [
                block_left + (block_right - block_left) * 0.42,
                block_bottom + pad_y,
                (block_right - block_left) * 0.56,
                (block_top - block_bottom) * 0.85,
            ]
        )
        ax_txt.axis("off")

        img_path = str(r["image_path"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize(img_size)
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center")

        vit = str(r.get("description_vit", r.get("description_vit-gpt2", r.get("description_vit", ""))))
        blip = str(r.get("description_blip", ""))
        vit_wrapped = "\n".join(textwrap.wrap(vit, width=wrap_width))
        blip_wrapped = "\n".join(textwrap.wrap(blip, width=wrap_width))

        title = f"ID: {r['id']} | {r.get('articleType','')}, {r.get('baseColour','')}"
        ax_txt.text(0.0, 1.02, title, fontsize=10, fontweight="bold", va="bottom")
        ax_txt.text(0.0, 0.78, "ViT-GPT2:", fontsize=9, fontweight="bold")
        ax_txt.text(0.0, 0.62, vit_wrapped, fontsize=9)
        ax_txt.text(0.0, 0.36, "BLIP:", fontsize=9, fontweight="bold")
        ax_txt.text(0.0, 0.20, blip_wrapped, fontsize=9)

        ax_border = fig.add_axes([block_left, block_bottom, block_right - block_left, block_top - block_bottom])
        ax_border.set_facecolor((1, 1, 1, 0))
        ax_border.set_xticks([])
        ax_border.set_yticks([])
        for spine in ax_border.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#DDDDDD")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
