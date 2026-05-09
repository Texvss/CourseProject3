# Fashion Captioning Pipeline

Refactored version of CourseProject-2.ipynb with a modular Python layout.

## Layout
- `pipeline.py` - thin runner that adds `src` to `PYTHONPATH`.
- `src/fashion_caption/config.py` - constants (seed, splits, prompt).
- `src/fashion_caption/preprocessing/` - dataset loading and splitting.
- `src/fashion_caption/models/` - caption models ViT-GPT2 and BLIP.
- `src/fashion_caption/models/blip2.py` - BLIP-2 (base by default, XL if you have a stronger GPU).
- `src/fashion_caption/eval/` - BLEU-1 metric and nltk download helper.
- `src/fashion_caption/visualization/` - comparison sheet PNG export.
- `data/` - expected to contain `styles.csv` and `images/`.
- `scripts/train_lora.py` - minimal LoRA trainer for BLIP-2.
- `scripts/serve_blip2_api.py` - remote BLIP-2 / BLIP-2 LoRA HTTP server for RunPod or another GPU box.
- `app.py` - Gradio UI with local BLIP / ViT-GPT2 and remote BLIP-2 / BLIP-2 LoRA modes.

## Install deps
```
pip install pandas numpy pillow torch torchvision transformers matplotlib tqdm nltk
```

Or install everything from the project file:
```
pip install -r requirements.txt
```

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

To keep large models off your laptop, run the BLIP-2 backend on RunPod:
```
PYTHONPATH=src python scripts/serve_blip2_api.py --device cuda --host 0.0.0.0 --port 8000
```

Then point `app.py` at that backend:
```
export FASHION_CAPTION_REMOTE_URL=http://<runpod-host>:8000/caption
export FASHION_CAPTION_REMOTE_MODEL_ID=Salesforce/blip2-opt-2.7b
export FASHION_CAPTION_REMOTE_ADAPTER_PATH=/workspace/lora-blip2-ecommerce
python app.py
```
