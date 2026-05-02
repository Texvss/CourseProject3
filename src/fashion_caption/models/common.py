from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    return Image.open(str(path)).convert("RGB")
