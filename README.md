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

## Install deps
```
pip install pandas numpy pillow torch torchvision transformers matplotlib tqdm nltk
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
