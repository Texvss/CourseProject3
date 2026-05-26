# Fashion Captioning Pipeline

Refactored version of CourseProject-2.ipynb with a modular Python layout.

## Layout
- `pipeline.py` - thin runner that adds `src` to `PYTHONPATH`.
- `src/fashion_caption/config.py` - constants (seed, splits, prompt).
- `src/fashion_caption/preprocessing/` - dataset loading and splitting.
- `src/fashion_caption/models/` - caption models ViT-GPT2 and BLIP.
- `src/fashion_caption/models/blip2.py` - BLIP-2 (base by default, XL if you have a stronger GPU).
- `src/fashion_caption/prompts.py` - shared e-commerce prompt templates.
- `src/fashion_caption/generation/` - unified generator registry for `vit-gpt2`, `blip`, `blip2`, `blip2-lora`, and `gpt`.
- `src/fashion_caption/postprocess/` - prompt echo cleanup, product-focus rewriting, color normalization, and length enforcement.
- `src/fashion_caption/eval/` - raw/cleaned BLEU-1, keyword metrics, Wilson CIs, protocol checks, and relaxed type matching.
- `src/fashion_caption/visualization/` - comparison sheet PNG export.
- `data/` - expected to contain `styles.csv` and `images/`.
- `scripts/train_lora.py` - minimal LoRA trainer for BLIP-2 with a style-only prompt.
- `scripts/regenerate_artifacts.py` - deterministic sanity/eval CSV and metrics generator.
- `scripts/runpod_artifacts.sh` - RunPod wrapper for BLIP-2 and BLIP-2+LoRA artifact generation.
- `scripts/serve_blip2_api.py` - remote BLIP-2 / BLIP-2 LoRA HTTP server plus `/api/generate/gpt`.
- `app.py` - dark Gradio studio with sidebar, saved runs, model selector, and GPT status tile.
- `.env.example` - optional environment variables for GPT, remote BLIP-2, and Gradio.

## Install deps
```
pip install pandas numpy pillow torch torchvision transformers matplotlib tqdm nltk
```

Or install everything from the project file:
```
pip install -r requirements.txt
```

`accelerate` is included for large Hugging Face models. `bitsandbytes` is optional and only installed from `requirements.txt` on compatible Linux x86_64 machines; it is not required for the local Gradio UI.

## Run
Option 1 (wrapper):
```
python3 pipeline.py --data-root data --out-dir outputs --device cpu
```

Option 2 (module):
```
PYTHONPATH=src python3 -m fashion_caption.cli --data-root data --out-dir outputs --device cpu
```

Flags:
- `--skip-vit` / `--skip-blip` - disable a model.
- `--skip-blip2` and `--blip2-model-id` - control BLIP-2.
- `--sanity-n` / `--eval-n` - split sizes.
- `--device` - cpu, cuda, mps or auto.

## Web UI
Run the local UI:
```
python app.py
```

The UI supports:
- Local `vit-gpt2` and `blip`.
- Remote `blip2` and `blip2-lora` through the `/caption` endpoint.
- GPT through OpenAI's Responses API only when both `ENABLE_GPT_API=1` and `OPENAI_API_KEY` are configured.

Optional environment setup:
```
cp .env.example .env
export ENABLE_GPT_API=1
export OPENAI_API_KEY=sk-...
export OPENAI_GPT_MODEL=gpt-4.1-mini
```

The app reads environment variables from the shell. It does not read `.env` automatically, so export the values or load them with your preferred shell tooling.

To keep large models off your laptop, run the BLIP-2 backend on RunPod:
```
PYTHONPATH=src python scripts/serve_blip2_api.py \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --quant "${BLIP2_QUANT:-none}"
```

Then point `app.py` at that backend:
```
export FASHION_CAPTION_REMOTE_URL=http://<runpod-host>:8000/caption
export FASHION_CAPTION_REMOTE_MODEL_ID=Salesforce/blip2-opt-2.7b
export FASHION_CAPTION_REMOTE_ADAPTER_PATH=/workspace/lora-blip2-ecommerce
python app.py
```

## GPT API endpoint
The FastAPI server exposes a GPT route:
```
PYTHONPATH=src python scripts/serve_blip2_api.py --host 0.0.0.0 --port 8000
```

POST multipart form data to:
```
POST /api/generate/gpt
```

Fields:
- `image` - required image file.
- `article_type` - optional dataset metadata.
- `base_colour` - optional dataset metadata.
- `prompt` - optional override/additional instruction.
- `openai_model` - optional model override; defaults to `OPENAI_GPT_MODEL` or `gpt-4.1-mini`.

If `OPENAI_API_KEY` is missing, the endpoint returns a helpful 400 response. Secrets are read only from environment variables and are not logged.
If `ENABLE_GPT_API` is not set to `1`, the endpoint is disabled even when a key exists.

To include GPT in quantitative artifact generation, pass the long flag `--include-gpt`; without it, GPT is excluded.

## Reproduce artifacts and metrics
Local CPU run for the lightweight models:
```
PYTHONPATH=src python scripts/regenerate_artifacts.py \
  --data-root data \
  --out-dir outputs \
  --device cpu \
  --models vit-gpt2,blip
```

RunPod GPU run for all non-GPT backends:
```
BLIP2_ADAPTER_PATH=/workspace/lora-blip2-ecommerce \
bash scripts/runpod_artifacts.sh
```

The artifact script writes `df_all_{sanity,eval}.csv`, per-model `df_*_{sanity,eval}.csv`, `metrics_{sanity,eval}.csv`, `artifact_status.json`, and metric plots into `outputs/`. The sanity split is for qualitative checks; headline metrics use the eval split (`n=300` by default).

Optional BLIP-2 quantization is controlled by `BLIP2_QUANT=none|8bit|4bit` or `--blip2-quant none|8bit|4bit`. Quantization is best used on CUDA Linux with `bitsandbytes`; if unavailable, the loader prints a warning and continues without quantization.

## Batch CSV export
Use the optional helper to export generation results:
```
python3 pipeline.py \
  --data-root data \
  --export-generation-csv outputs/generation_export.csv \
  --export-models vit-gpt2,blip \
  --export-n 20
```

The CSV contains:
`id,image_path,articleType,baseColour,productDisplayName,model_id,prompt_id,prompt,raw_output,cleaned_output,generated_text,error`

For remote BLIP-2 in the export:
```
python3 pipeline.py \
  --data-root data \
  --export-generation-csv outputs/blip2_export.csv \
  --export-models blip2 \
  --remote-url http://<runpod-host>:8000/caption
```

Add `--include-metrics` to print BLEU-1 when references are available.
Metrics CSVs report raw rows with strict/relaxed type and color keyword rates plus Wilson 95% CIs. Cleaned rows omit type/color keyword metrics as `omitted_by_construction` because the cleaner may insert metadata; they instead report service metrics such as protocol compliance, forbidden mentions, and prompt echo.

## Smoke tests
```
PYTHONPATH=src python scripts/smoke_test.py
```
