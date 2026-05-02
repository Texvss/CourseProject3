"""
Gradio demo for captioning with ViT-GPT2 or BLIP.
Run: source .venv/bin/activate && python app.py
"""
import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption import config  # noqa: E402
from fashion_caption.models import vitgpt2, blip  # noqa: E402
from fashion_caption.postprocess.text import enforce_type_color  # noqa: E402


device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

MODEL_CACHE = {}


def get_model(name: str):
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    if name == "ViT-GPT2":
        MODEL_CACHE[name] = vitgpt2
    elif name == "BLIP":
        MODEL_CACHE[name] = blip
    else:
        raise ValueError("Unknown model")
    return MODEL_CACHE[name]


def caption_image(image, model_name):
    if image is None:
        return "Upload an image first."
    model_mod = get_model(model_name)
    # build a tiny one-row dataframe-like object
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "id": "demo",
                "image_path": image,
                "articleType": "",
                "baseColour": "",
            }
        ]
    )
    if model_name == "ViT-GPT2":
        res = model_mod.caption(df, device=device)
    else:
        res = model_mod.caption(df, device=device)
    text = res.iloc[0]["description"]
    text_pp = enforce_type_color(text, res.iloc[0].get("articleType"), res.iloc[0].get("baseColour"))
    return text_pp


demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="filepath", label="Upload a garment photo"),
        gr.Radio(["BLIP", "ViT-GPT2"], value="BLIP", label="Model"),
    ],
    outputs=gr.Textbox(label="Generated description"),
    title="Fashion Captioning Demo",
    description="Generates a concise e-commerce description of the garment.",
)

if __name__ == "__main__":
    demo.launch()
