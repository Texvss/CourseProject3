"""
Gradio demo for local BLIP / ViT-GPT2 and remote BLIP-2 / BLIP-2 LoRA.

Local:
    source .venv/bin/activate && python app.py

Remote backend (RunPod or another GPU box):
    PYTHONPATH=src python scripts/serve_blip2_api.py --device cuda --host 0.0.0.0 --port 8000
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption.models import vitgpt2, blip  # noqa: E402
from fashion_caption.postprocess.text import enforce_type_color  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_CACHE = {}
REMOTE_URL_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_URL", "http://127.0.0.1:8000/caption")
REMOTE_MODEL_ID_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_MODEL_ID", "Salesforce/blip2-opt-2.7b")
REMOTE_ADAPTER_PATH_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_ADAPTER_PATH", "")


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


def caption_local_image(image: str, model_name: str) -> str:
    model_mod = get_model(model_name)
    df = pd.DataFrame(
        [{"id": "demo", "image_path": image, "articleType": "", "baseColour": ""}]
    )
    res = model_mod.caption(df, device=device)
    text = res.iloc[0]["description"]
    return enforce_type_color(text, res.iloc[0].get("articleType"), res.iloc[0].get("baseColour"))


def caption_remote_image(
    image_path: str,
    remote_url: str,
    model_id: str,
    adapter_path: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    with open(image_path, "rb") as fh:
        image_bytes = fh.read()

    boundary = "----fashion-caption-boundary"
    parts = []

    def add_field(name: str, value: str):
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    add_field("model_id", model_id)
    add_field("adapter_path", adapter_path)
    add_field("max_new_tokens", str(max_new_tokens))
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="upload.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    body = b"".join(parts)
    request = urllib.request.Request(
        remote_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Remote server returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach remote BLIP-2 backend: {exc.reason}") from exc

    return payload["description"], json.dumps(payload, ensure_ascii=False, indent=2)


def caption_image(image, model_name, remote_url, remote_model_id, remote_adapter_path, max_new_tokens):
    if image is None:
        return "Upload an image first.", ""
    if model_name in {"BLIP", "ViT-GPT2"}:
        description = caption_local_image(image, model_name)
        metadata = {
            "mode": "local",
            "device": str(device),
            "model_name": model_name,
        }
        return description, json.dumps(metadata, ensure_ascii=False, indent=2)

    adapter_path = remote_adapter_path if model_name == "BLIP-2 LoRA (remote)" else ""
    description, payload = caption_remote_image(
        image_path=image,
        remote_url=remote_url,
        model_id=remote_model_id,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
    )
    return description, payload


with gr.Blocks(title="Fashion Captioning Demo") as demo:
    gr.Markdown(
        """
        # Fashion Captioning Demo
        Local BLIP / ViT-GPT2 run on this machine. BLIP-2 and BLIP-2 LoRA call a remote GPU backend
        (for example RunPod) so you do not need to keep heavy checkpoints on your laptop.
        """
    )
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload a garment photo")
        with gr.Column():
            model_input = gr.Radio(
                ["BLIP", "ViT-GPT2", "BLIP-2 (remote)", "BLIP-2 LoRA (remote)"],
                value="BLIP",
                label="Model",
            )
            max_new_tokens_input = gr.Slider(
                minimum=16,
                maximum=128,
                step=4,
                value=60,
                label="Max new tokens",
            )
            remote_url_input = gr.Textbox(
                value=REMOTE_URL_DEFAULT,
                label="Remote BLIP-2 endpoint",
                info="Example: http://<runpod-host>:8000/caption",
            )
            remote_model_id_input = gr.Textbox(
                value=REMOTE_MODEL_ID_DEFAULT,
                label="Remote BLIP-2 model id",
            )
            remote_adapter_path_input = gr.Textbox(
                value=REMOTE_ADAPTER_PATH_DEFAULT,
                label="Remote LoRA adapter path",
                info="Used only for BLIP-2 LoRA (remote). Example: /workspace/lora-blip2-ecommerce",
            )
            submit_btn = gr.Button("Generate")

    description_output = gr.Textbox(label="Generated description", lines=4)
    metadata_output = gr.Code(label="Backend response / metadata", language="json")

    submit_btn.click(
        fn=caption_image,
        inputs=[
            image_input,
            model_input,
            remote_url_input,
            remote_model_id_input,
            remote_adapter_path_input,
            max_new_tokens_input,
        ],
        outputs=[description_output, metadata_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
