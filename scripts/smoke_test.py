from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_caption.eval import batch
from fashion_caption.generation.base import GenerationResult
from fashion_caption.models import blip2
from fashion_caption.postprocess.text import clean_description
from fashion_caption.prompts import get_prompt_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small smoke checks for the fashion captioning repo.")
    parser.add_argument("--data-root", default="data", help="Optional data root; not required for fake export checks.")
    return parser.parse_args()


def check_postprocess() -> None:
    out = clean_description("description: a model wearing a red printed top in front of a wall", article_type="Tops", base_colour="Red")
    assert "red" in out.lower()
    assert "top" in out.lower()
    print(f"[smoke] OK: postprocess -> {out}")


class _FakeConfig:
    use_decoder_only_language_model = True


class _FakeBlip2Model:
    config = _FakeConfig()


def check_blip2_prompt_echo_removal() -> None:
    inputs = {"input_ids": torch.tensor([[10, 11, 12]])}
    output_ids = torch.tensor([[10, 11, 12, 20, 21]])
    prompt_len = blip2._decoder_prompt_length(inputs, output_ids, _FakeBlip2Model())
    assert prompt_len == 3
    assert output_ids[0][prompt_len:].tolist() == [20, 21]
    print("[smoke] OK: BLIP-2 prompt prefix is sliced")


def check_blip2_outputs_can_differ() -> None:
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    first = torch.tensor([[1, 2, 3, 7, 8]])
    second = torch.tensor([[1, 2, 3, 9, 10]])
    prompt_len = blip2._decoder_prompt_length(inputs, first, _FakeBlip2Model())
    assert first[0][prompt_len:].tolist() != second[0][prompt_len:].tolist()
    print("[smoke] OK: different completions remain distinguishable")


class _FakeRegistry:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, model_id, image, prompt_config, params=None):
        return GenerationResult(
            text="A blue T-shirt.",
            meta={
                "model_id": model_id,
                "raw_description": "a person wearing a blue shirt",
                "raw_output": "a person wearing a blue shirt",
            },
        )


def check_csv_export_schema() -> None:
    original_registry = batch.ModelRegistry
    batch.ModelRegistry = _FakeRegistry
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sample.jpg"
            Image.new("RGB", (16, 16), color=(50, 80, 180)).save(image_path)
            df = pd.DataFrame(
                [
                    {
                        "id": "1",
                        "image_path": image_path,
                        "articleType": "Tshirts",
                        "baseColour": "Blue",
                        "productDisplayName": "Blue T-shirt",
                    }
                ]
            )
            out_csv = tmp_path / "export.csv"
            batch.export_generation_csv(df, out_csv, ["vit-gpt2"], fail_fast=True)
            exported = pd.read_csv(out_csv)
            assert not exported.empty
            for col in ["raw_output", "cleaned_output", "error"]:
                assert col in exported.columns
            assert exported.loc[0, "raw_output"]
            assert exported.loc[0, "cleaned_output"]
        print("[smoke] OK: CSV export has raw/clean/error")
    finally:
        batch.ModelRegistry = original_registry


def check_gpt_route_disabled() -> None:
    os.environ.pop("ENABLE_GPT_API", None)
    os.environ.pop("OPENAI_API_KEY", None)
    old_argv = sys.argv[:]
    try:
        sys.argv = ["serve_blip2_api.py"]
        from fastapi.testclient import TestClient

        module_path = ROOT / "scripts" / "serve_blip2_api.py"
        spec = importlib.util.spec_from_file_location("serve_blip2_api_smoke", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color=(200, 40, 40)).save(buffer, format="PNG")
        response = TestClient(module.app).post(
            "/api/generate/gpt",
            files={"image": ("sample.png", buffer.getvalue(), "image/png")},
            data={"prompt": get_prompt_config().prompt},
        )
        assert response.status_code == 400
        assert "disabled" in response.text.lower()
        print("[smoke] OK: GPT route disabled without feature flag")
    finally:
        sys.argv = old_argv


def main() -> None:
    parse_args()
    check_postprocess()
    check_blip2_prompt_echo_removal()
    check_blip2_outputs_can_differ()
    check_csv_export_schema()
    check_gpt_route_disabled()


if __name__ == "__main__":
    main()
