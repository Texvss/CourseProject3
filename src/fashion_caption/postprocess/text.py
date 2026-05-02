import re
from typing import Optional

COLOR_WORDS = {
    "black",
    "white",
    "grey",
    "gray",
    "red",
    "green",
    "blue",
    "navy",
    "pink",
    "purple",
    "yellow",
    "orange",
    "brown",
    "beige",
    "maroon",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ensure_token(text: str, token: str) -> str:
    """
    If token (case-insensitive) is not in text, prepend it.
    """
    if not token:
        return text
    if normalize(token) in normalize(text):
        return text
    return f"{token.strip()} - {text}"


def enforce_type_color(description: str, article_type: Optional[str], base_colour: Optional[str]) -> str:
    """
    Ensure garment type and main color appear in the description.
    """
    out = description.strip()
    # enforce type
    out = ensure_token(out, article_type or "")
    # enforce color (simple heuristic: if base_colour not present and looks like a color word)
    if base_colour:
        color_norm = normalize(base_colour)
        has_color_word = any(c in normalize(out) for c in COLOR_WORDS)
        if color_norm not in normalize(out) and (color_norm in COLOR_WORDS or not has_color_word):
            out = ensure_token(out, base_colour)
    return out
