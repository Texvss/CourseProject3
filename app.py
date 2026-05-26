"""
Gradio studio for local BLIP / ViT-GPT2, remote BLIP-2 / BLIP-2 LoRA, and GPT.

Run locally:
    source .venv/bin/activate && python app.py

Remote BLIP-2 backend:
    PYTHONPATH=src python scripts/serve_blip2_api.py --device cuda --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import gradio as gr
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption.generation import ModelRegistry  # noqa: E402
from fashion_caption.generation.openai_api import DEFAULT_OPENAI_MODEL, gpt_feature_enabled, openai_key_configured  # noqa: E402
from fashion_caption.generation.registry import OpenAIConfigurationError, OpenAIGenerationError  # noqa: E402
from fashion_caption.prompts import ECOMMERCE_PROMPT_ID, get_prompt_config  # noqa: E402


device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
REGISTRY = ModelRegistry(device=device)

REMOTE_URL_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_URL", "http://127.0.0.1:8000/caption")
REMOTE_MODEL_ID_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_MODEL_ID", "Salesforce/blip2-opt-2.7b")
REMOTE_ADAPTER_PATH_DEFAULT = os.environ.get("FASHION_CAPTION_REMOTE_ADAPTER_PATH", "")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_GPT_MODEL", DEFAULT_OPENAI_MODEL)

MODEL_CHOICES = ["vit-gpt2", "blip", "blip2", "blip2-lora"]
if REGISTRY.gpt_configured():
    MODEL_CHOICES.append("gpt")

PAGE_CSS = """
:root {
    --bg: #101010;
    --panel: #181818;
    --panel-soft: #202020;
    --line: #303030;
    --line-soft: #262626;
    --text: #f2f0eb;
    --muted: #a6a19a;
    --accent: #d7ff70;
    --accent-2: #7dd3c7;
    --danger: #ff8b8b;
}
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.studio-shell {
    min-height: calc(100vh - 32px);
    gap: 0 !important;
}
.sidebar {
    background: #151515;
    border-right: 1px solid var(--line);
    min-width: 250px !important;
    padding: 18px 14px !important;
}
.main-pane {
    min-height: calc(100vh - 32px);
    padding: 28px 32px 18px !important;
}
.sidebar-title {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 18px 0 8px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #d8d3ca;
    border-radius: 10px;
    padding: 9px 10px;
    margin: 2px 0;
    background: transparent;
    border: 1px solid transparent;
}
.nav-item.active {
    background: #222;
    border-color: var(--line);
}
.landing {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 38vh;
    text-align: center;
}
.landing h1 {
    color: var(--text);
    font-size: 2.25rem;
    line-height: 1.12;
    margin: 0 0 12px;
    letter-spacing: 0;
}
.landing p {
    color: var(--muted);
    margin: 0;
    max-width: 620px;
    line-height: 1.55;
}
.preview-panel,
.output-panel,
.composer,
.status-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
}
.preview-panel,
.output-panel,
.status-panel {
    padding: 16px;
}
.composer {
    position: sticky;
    bottom: 12px;
    padding: 12px;
    margin-top: 18px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.35);
}
.run-card {
    background: var(--panel-soft);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}
.run-card h3,
.gpt-tile h3 {
    color: var(--text);
    font-size: 0.96rem;
    margin: 0 0 8px;
    letter-spacing: 0;
}
.run-card p {
    color: var(--text);
    margin: 0;
    line-height: 1.45;
}
.meta-row {
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.82rem;
    margin-top: 10px;
}
.pill,
.status-ok,
.status-warn,
.status-bad {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 5px 9px;
    font-size: 0.78rem;
    line-height: 1;
    border: 1px solid var(--line);
}
.pill { color: var(--muted); background: #171717; }
.status-ok { color: #dafeb7; background: rgba(124, 184, 77, 0.16); border-color: rgba(215, 255, 112, 0.24); }
.status-warn { color: #ffe0a3; background: rgba(255, 190, 96, 0.12); border-color: rgba(255, 190, 96, 0.22); }
.status-bad { color: var(--danger); background: rgba(255, 93, 93, 0.12); border-color: rgba(255, 93, 93, 0.22); }
.gpt-tile {
    background: #1c1c1c;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
}
.gpt-tile p {
    color: var(--muted);
    margin: 6px 0 0;
}
.gradio-container button.primary {
    background: var(--accent) !important;
    color: #111 !important;
    border: 0 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #121212 !important;
    color: var(--text) !important;
}
@media (max-width: 900px) {
    .studio-shell {
        flex-direction: column !important;
    }
    .sidebar {
        min-width: 0 !important;
        border-right: 0;
        border-bottom: 1px solid var(--line);
    }
    .main-pane {
        padding: 18px 14px !important;
    }
    .composer {
        position: static;
    }
}
"""


def build_status(kind: str, text: str) -> str:
    cls = {"ok": "status-ok", "warn": "status-warn", "bad": "status-bad"}.get(kind, "status-warn")
    return f'<span class="{cls}">{text}</span>'


def landing_html() -> str:
    return """
    <div class="landing">
        <h1>С чего начнем?</h1>
        <p>Upload a product image, choose a model, and generate short catalog-ready copy focused on item type and color.</p>
    </div>
    """


def metadata_for_image(image_path: Optional[str]) -> Dict[str, str]:
    if not image_path:
        return {"articleType": "", "baseColour": ""}
    image_id = Path(image_path).stem
    styles_path = ROOT / "data" / "styles.csv"
    if not styles_path.exists():
        return {"articleType": "", "baseColour": ""}
    try:
        with styles_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("id", "")) == image_id:
                    return {
                        "articleType": row.get("articleType", "") or "",
                        "baseColour": row.get("baseColour", "") or "",
                    }
    except UnicodeDecodeError:
        with styles_path.open(newline="", encoding="latin-1") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("id", "")) == image_id:
                    return {
                        "articleType": row.get("articleType", "") or "",
                        "baseColour": row.get("baseColour", "") or "",
                    }
    return {"articleType": "", "baseColour": ""}


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


def format_usage(usage: Dict[str, Any]) -> str:
    if not usage:
        return "tokens unavailable"
    input_tokens = usage.get("input_tokens", "?")
    output_tokens = usage.get("output_tokens", "?")
    total_tokens = usage.get("total_tokens", "?")
    return f"{input_tokens} in / {output_tokens} out / {total_tokens} total"


def gpt_tile_html(record: Optional[Dict[str, Any]] = None) -> str:
    configured = REGISTRY.gpt_configured()
    if configured:
        status = build_status("ok", "GPT enabled")
    elif not gpt_feature_enabled():
        status = build_status("warn", "GPT disabled")
    elif not openai_key_configured():
        status = build_status("bad", "OPENAI_API_KEY missing")
    else:
        status = build_status("bad", "GPT unavailable")
    model = OPENAI_MODEL_DEFAULT
    details = f"<p>Default model: <strong>{model}</strong></p>"
    if record and record.get("model_id") == "gpt":
        meta = record.get("meta", {})
        usage = format_usage(meta.get("usage") or {})
        latency = meta.get("latency_ms", "n/a")
        details += f"<p>{usage}</p><p>Latency: {latency} ms</p>"
    return f"""
    <div class="gpt-tile">
        <h3>GPT generator</h3>
        {status}
        {details}
    </div>
    """


def run_card_html(record: Dict[str, Any]) -> str:
    meta = record.get("meta", {})
    prompt_id = record.get("prompt_id", ECOMMERCE_PROMPT_ID)
    latency = meta.get("latency_ms", "n/a")
    article_type = record.get("articleType") or "unknown type"
    base_colour = record.get("baseColour") or "unknown color"
    return f"""
    <div class="run-card">
        <h3>{record.get("model_id", "model")} · {prompt_id}</h3>
        <p>{record.get("text", "")}</p>
        <div class="meta-row">
            <span class="pill">{article_type}</span>
            <span class="pill">{base_colour}</span>
            <span class="pill">{latency} ms</span>
        </div>
    </div>
    """


def output_cards_html(records: List[Dict[str, Any]]) -> str:
    if not records:
        return '<div class="run-card"><p>No generations yet.</p></div>'
    return "\n".join(run_card_html(record) for record in records[:6])


def saved_run_choices(records: List[Dict[str, Any]]) -> List[str]:
    choices = []
    for idx, record in enumerate(records[:12], start=1):
        text = record.get("text", "")
        short_text = text[:48] + ("..." if len(text) > 48 else "")
        choices.append(f"{idx}. {record.get('model_id', '')} · {short_text}")
    return choices


def generate_description(
    image_path: Optional[str],
    prompt_text: str,
    model_id: str,
    max_new_tokens: int,
    remote_url: str,
    remote_model_id: str,
    remote_adapter_path: str,
    openai_model: str,
    records: Optional[List[Dict[str, Any]]],
):
    records = list(records or [])
    if not image_path:
        status = build_status("warn", "Upload a product image first.")
        choices = saved_run_choices(records)
        return landing_html(), output_cards_html(records), gpt_tile_html(), gr.update(choices=choices, value=choices[0] if choices else None), status, "", records, None

    metadata = metadata_for_image(image_path)
    prompt_config = get_prompt_config(
        prompt_id=ECOMMERCE_PROMPT_ID,
        article_type=metadata.get("articleType"),
        base_colour=metadata.get("baseColour"),
        user_prompt=prompt_text,
    )

    params: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "max_words": 25,
    }
    if model_id in {"blip2", "blip2-lora"}:
        params.update(
            {
                "remote_url": remote_url,
                "hf_model_id": remote_model_id,
                "adapter_path": remote_adapter_path if model_id == "blip2-lora" else "",
            }
        )
    if model_id == "gpt":
        if not REGISTRY.gpt_configured():
            status = build_status("bad", "GPT is disabled. Set ENABLE_GPT_API=1 and OPENAI_API_KEY to enable it.")
            choices = saved_run_choices(records)
            return landing_html(), output_cards_html(records), gpt_tile_html(), gr.update(choices=choices, value=choices[0] if choices else None), status, "", records, image_path
        params.update({"openai_model": openai_model or OPENAI_MODEL_DEFAULT})

    try:
        image = Image.open(image_path).convert("RGB")
        result = REGISTRY.generate(
            model_id=model_id,
            image=image,
            prompt_config=prompt_config,
            params=params,
        )
    except OpenAIConfigurationError as exc:
        status = build_status("bad", str(exc))
        choices = saved_run_choices(records)
        return landing_html(), output_cards_html(records), gpt_tile_html(), gr.update(choices=choices, value=choices[0] if choices else None), status, "", records, image_path
    except OpenAIGenerationError as exc:
        status = build_status("bad", str(exc))
        choices = saved_run_choices(records)
        return landing_html(), output_cards_html(records), gpt_tile_html(), gr.update(choices=choices, value=choices[0] if choices else None), status, "", records, image_path
    except Exception as exc:
        status = build_status("bad", str(exc))
        choices = saved_run_choices(records)
        return landing_html(), output_cards_html(records), gpt_tile_html(), gr.update(choices=choices, value=choices[0] if choices else None), status, "", records, image_path

    record = {
        "run_id": len(records) + 1,
        "model_id": model_id,
        "prompt_id": prompt_config.prompt_id,
        "text": result.text,
        "meta": result.meta,
        "image_path": image_path,
        "articleType": metadata.get("articleType", ""),
        "baseColour": metadata.get("baseColour", ""),
    }
    records.insert(0, record)
    status = build_status("ok", f"{model_id} generated an e-commerce description.")
    metadata_json = json.dumps(result.meta, ensure_ascii=False, indent=2)
    choices = saved_run_choices(records)
    return (
        run_card_html(record),
        output_cards_html(records),
        gpt_tile_html(record),
        gr.update(choices=choices, value=choices[0] if choices else None),
        status,
        metadata_json,
        records,
        image_path,
    )


def select_saved_run(records: List[Dict[str, Any]], selected: Optional[str]):
    records = list(records or [])
    if not selected:
        return landing_html(), output_cards_html(records), gpt_tile_html(), "", None
    try:
        row_index = int(selected.split(".", 1)[0]) - 1
    except (ValueError, IndexError):
        row_index = 0
    if row_index < 0 or row_index >= len(records):
        return landing_html(), output_cards_html(records), gpt_tile_html(), "", None
    record = records[row_index]
    return (
        run_card_html(record),
        output_cards_html(records),
        gpt_tile_html(record),
        json.dumps(record.get("meta", {}), ensure_ascii=False, indent=2),
        record.get("image_path"),
    )


def new_generation():
    return [], gr.update(choices=[], value=None), landing_html(), output_cards_html([]), gpt_tile_html(), build_status("warn", "Upload an image and choose a model."), "", None


def check_remote(remote_url: str):
    try:
        payload, payload_json = fetch_remote_health(remote_url)
        label = build_status("ok", f"Remote backend is ready on {payload.get('device', 'unknown device')}.")
        return label, payload_json
    except Exception as exc:
        return build_status("bad", f"Remote backend check failed: {exc}"), ""


def toggle_model_fields(model_id: str):
    is_remote = model_id in {"blip2", "blip2-lora"}
    needs_lora = model_id == "blip2-lora"
    is_gpt = model_id == "gpt"
    return (
        gr.update(visible=is_remote),
        gr.update(visible=is_remote),
        gr.update(visible=needs_lora),
        gr.update(visible=is_remote),
        gr.update(visible=is_gpt),
    )


with gr.Blocks(title="Fashion Captioning Studio") as demo:
    run_state = gr.State([])
    with gr.Row(elem_classes=["studio-shell"]):
        with gr.Column(scale=2, elem_classes=["sidebar"]):
            new_btn = gr.Button("New generation", variant="primary")
            gr.HTML(
                """
                <div class="sidebar-title">Navigation</div>
                <div class="nav-item active">Projects</div>
                <div class="nav-item">Recent</div>
                <div class="nav-item">Evaluations</div>
                <div class="nav-item">Settings</div>
                <div class="sidebar-title">Saved runs</div>
                """
            )
            saved_runs = gr.Radio(
                choices=[],
                interactive=True,
                label=None,
            )
            gpt_tile = gr.HTML(gpt_tile_html())

        with gr.Column(scale=7, elem_classes=["main-pane"]):
            status_output = gr.HTML(build_status("warn", "Upload an image and choose a model."))
            with gr.Row():
                with gr.Column(scale=4, elem_classes=["preview-panel"]):
                    image_input = gr.Image(
                        type="filepath",
                        label="Product image",
                        height=320,
                    )
                    gr.Examples(
                        examples=[
                            ["data/images/15970.jpg"],
                            ["data/images/30805.jpg"],
                            ["data/images/31432.jpg"],
                        ],
                        inputs=image_input,
                        label="Dataset examples",
                    )
                with gr.Column(scale=6, elem_classes=["output-panel"]):
                    run_view = gr.HTML(landing_html())
                    output_cards = gr.HTML(output_cards_html([]))
                    with gr.Accordion("Backend response / diagnostics", open=False):
                        metadata_output = gr.Code(label="Response / metadata", language="json")

            with gr.Row(elem_classes=["composer"]):
                prompt_input = gr.Textbox(
                    label=None,
                    placeholder="Optional extra instruction for this generation",
                    lines=1,
                    scale=5,
                )
                model_input = gr.Dropdown(
                    MODEL_CHOICES,
                    value="blip",
                    label=None,
                    scale=2,
                )
                max_new_tokens_input = gr.Slider(
                    minimum=16,
                    maximum=128,
                    step=4,
                    value=60,
                    label="Tokens",
                    scale=2,
                )
                submit_btn = gr.Button("Generate", variant="primary", scale=1)

            with gr.Row():
                remote_url_input = gr.Textbox(
                    value=REMOTE_URL_DEFAULT,
                    label="Remote BLIP-2 endpoint",
                    visible=False,
                )
                remote_model_id_input = gr.Textbox(
                    value=REMOTE_MODEL_ID_DEFAULT,
                    label="Remote BLIP-2 model id",
                    visible=False,
                )
            with gr.Row():
                remote_adapter_path_input = gr.Textbox(
                    value=REMOTE_ADAPTER_PATH_DEFAULT,
                    label="Remote LoRA adapter path",
                    visible=False,
                )
                check_remote_btn = gr.Button("Check remote backend", variant="secondary", visible=False)
                openai_model_input = gr.Textbox(
                    value=OPENAI_MODEL_DEFAULT,
                    label="OpenAI model",
                    visible=False,
                )

    submit_btn.click(
        fn=generate_description,
        inputs=[
            image_input,
            prompt_input,
            model_input,
            max_new_tokens_input,
            remote_url_input,
            remote_model_id_input,
            remote_adapter_path_input,
            openai_model_input,
            run_state,
        ],
        outputs=[
            run_view,
            output_cards,
            gpt_tile,
            saved_runs,
            status_output,
            metadata_output,
            run_state,
            image_input,
        ],
    )
    saved_runs.change(
        fn=select_saved_run,
        inputs=[run_state, saved_runs],
        outputs=[run_view, output_cards, gpt_tile, metadata_output, image_input],
    )
    new_btn.click(
        fn=new_generation,
        inputs=[],
        outputs=[
            run_state,
            saved_runs,
            run_view,
            output_cards,
            gpt_tile,
            status_output,
            metadata_output,
            image_input,
        ],
    )
    check_remote_btn.click(
        fn=check_remote,
        inputs=[remote_url_input],
        outputs=[status_output, metadata_output],
    )
    model_input.change(
        fn=toggle_model_fields,
        inputs=[model_input],
        outputs=[
            remote_url_input,
            remote_model_id_input,
            remote_adapter_path_input,
            check_remote_btn,
            openai_model_input,
        ],
    )


if __name__ == "__main__":
    server_port_env = os.environ.get("GRADIO_SERVER_PORT")
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(server_port_env) if server_port_env else None,
        css=PAGE_CSS,
    )
