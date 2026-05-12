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
from urllib.parse import urlparse
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
PAGE_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at 20% 20%, rgba(255, 166, 0, 0.12), transparent 22%),
        radial-gradient(circle at 80% 0%, rgba(0, 194, 255, 0.12), transparent 24%),
        linear-gradient(180deg, #0e1014 0%, #151922 48%, #0f131b 100%);
}
.app-shell {
    max-width: 1240px;
    margin: 0 auto;
}
.hero {
    padding: 20px 22px 12px 22px;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border-radius: 22px;
    box-shadow: 0 18px 60px rgba(0,0,0,0.28);
}
.hero h1 {
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: -0.03em;
}
.hero p {
    margin: 10px 0 0;
    opacity: 0.9;
    line-height: 1.6;
}
.metric-card {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    padding: 14px 16px;
}
.metric-card strong {
    display: block;
    font-size: 0.82rem;
    opacity: 0.72;
    margin-bottom: 4px;
}
.metric-card span {
    font-size: 1.1rem;
}
.panel {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    background: rgba(12, 15, 23, 0.75);
    box-shadow: 0 18px 48px rgba(0,0,0,0.22);
}
.section-title {
    font-size: 1rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 8px;
}
.status-ok, .status-warn, .status-bad {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 600;
}
.status-ok { background: rgba(60, 194, 104, 0.18); color: #94f0ae; }
.status-warn { background: rgba(255, 170, 70, 0.16); color: #ffd28a; }
.status-bad { background: rgba(255, 95, 95, 0.16); color: #ff9b9b; }
"""


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


def build_status(kind: str, text: str) -> str:
    cls = {
        "ok": "status-ok",
        "warn": "status-warn",
        "bad": "status-bad",
    }.get(kind, "status-warn")
    return f'<span class="{cls}">{text}</span>'


def remote_health_url(remote_url: str) -> str:
    if remote_url.rstrip("/").endswith("/caption"):
        return remote_url.rsplit("/", 1)[0] + "/health"
    parsed = urlparse(remote_url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else remote_url.rstrip("/")
    return base + "/health"


def fetch_remote_health(remote_url: str) -> tuple[dict, str]:
    request = urllib.request.Request(remote_health_url(remote_url), method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, json.dumps(payload, ensure_ascii=False, indent=2)


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
        return "", build_status("warn", "Upload a garment photo to start."), ""
    try:
        if model_name in {"BLIP", "ViT-GPT2"}:
            description = caption_local_image(image, model_name)
            metadata = {
                "mode": "local",
                "device": str(device),
                "model_name": model_name,
            }
            status = build_status("ok", f"{model_name} generated a local caption on {device}.")
            if not description.strip():
                status = build_status("warn", f"{model_name} returned an empty caption. Try another product image.")
            return description, status, json.dumps(metadata, ensure_ascii=False, indent=2)

        if model_name == "BLIP-2 LoRA (remote)" and not remote_adapter_path.strip():
            return "", build_status("bad", "LoRA mode selected, but the adapter path is empty."), ""

        adapter_path = remote_adapter_path if model_name == "BLIP-2 LoRA (remote)" else ""
        description, payload = caption_remote_image(
            image_path=image,
            remote_url=remote_url,
            model_id=remote_model_id,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        mode_label = "BLIP-2 LoRA" if adapter_path else "BLIP-2"
        status = build_status("ok", f"{mode_label} responded from the remote GPU backend.")
        if not description.strip():
            status = build_status(
                "warn",
                "The backend responded, but the model returned an empty description. Try another image or increase max tokens.",
            )
        return description, status, payload
    except Exception as exc:
        return "", build_status("bad", str(exc)), ""


def check_remote(remote_url: str):
    try:
        payload, payload_json = fetch_remote_health(remote_url)
        label = build_status("ok", f"Remote backend is ready on {payload.get('device', 'unknown device')}.")
        return label, payload_json
    except Exception as exc:
        return build_status("bad", f"Remote backend check failed: {exc}"), ""


def toggle_remote_fields(model_name: str):
    is_remote = "remote" in model_name.lower()
    needs_lora = model_name == "BLIP-2 LoRA (remote)"
    return (
        gr.update(visible=is_remote),
        gr.update(visible=is_remote),
        gr.update(visible=is_remote),
        gr.update(visible=needs_lora),
        gr.update(visible=is_remote),
    )


with gr.Blocks(title="Fashion Captioning Studio") as demo:
    gr.HTML('<div class="app-shell">')
    gr.Markdown(
        """
        <div class="hero">
            <h1>Fashion Captioning Studio</h1>
            <p>
                Generate e-commerce descriptions with lightweight local models or route heavy BLIP-2 / LoRA runs
                through a remote GPU backend. This demo is designed for quick side-by-side testing while keeping
                the laptop clean from giant checkpoints.
            </p>
        </div>
        """
    )
    gr.HTML(
        """
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 20px 0;">
            <div class="metric-card"><strong>Local models</strong><span>BLIP · ViT-GPT2</span></div>
            <div class="metric-card"><strong>Remote models</strong><span>BLIP-2 · BLIP-2 LoRA</span></div>
            <div class="metric-card"><strong>Recommended input</strong><span>Single product image on a clean background</span></div>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.Markdown("### Product image")
            image_input = gr.Image(type="filepath", label="Upload a garment photo", height=470)
            gr.Examples(
                examples=[
                    ["data/images/15970.jpg"],
                    ["data/images/30805.jpg"],
                    ["data/images/31432.jpg"],
                ],
                inputs=image_input,
                label="Quick examples from the dataset",
            )
        with gr.Column(scale=6, elem_classes=["panel"]):
            gr.Markdown("### Inference controls")
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
                visible=False,
            )
            remote_model_id_input = gr.Textbox(
                value=REMOTE_MODEL_ID_DEFAULT,
                label="Remote BLIP-2 model id",
                visible=False,
            )
            remote_adapter_path_input = gr.Textbox(
                value=REMOTE_ADAPTER_PATH_DEFAULT,
                label="Remote LoRA adapter path",
                info="Used only for BLIP-2 LoRA (remote). Example: /workspace/lora-blip2-ecommerce",
                visible=False,
            )
            with gr.Row(visible=False) as remote_actions_row:
                check_remote_btn = gr.Button("Check remote backend", variant="secondary")
            submit_btn = gr.Button("Generate description", variant="primary")

    status_output = gr.HTML(value=build_status("warn", "Choose a model and image, then generate a caption."))
    description_output = gr.Textbox(label="Generated description", lines=5, placeholder="Generated catalog copy will appear here.")
    with gr.Accordion("Backend response / diagnostics", open=False):
        metadata_output = gr.Code(label="Response / metadata", language="json")

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
        outputs=[description_output, status_output, metadata_output],
    )
    check_remote_btn.click(
        fn=check_remote,
        inputs=[remote_url_input],
        outputs=[status_output, metadata_output],
    )
    model_input.change(
        fn=toggle_remote_fields,
        inputs=[model_input],
        outputs=[
            remote_url_input,
            remote_model_id_input,
            remote_actions_row,
            remote_adapter_path_input,
            check_remote_btn,
        ],
    )
    gr.HTML("</div>")

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        css=PAGE_CSS,
    )
