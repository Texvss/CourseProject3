from __future__ import annotations

import re
from typing import Iterable, Optional

from fashion_caption.prompts import ECOMMERCE_PROTOCOL


DATASET_COLOURS = {
    "beige": "Beige",
    "black": "Black",
    "blue": "Blue",
    "bronze": "Bronze",
    "brown": "Brown",
    "burgundy": "Burgundy",
    "charcoal": "Charcoal",
    "coffee brown": "Coffee Brown",
    "copper": "Copper",
    "cream": "Cream",
    "fluorescent green": "Fluorescent Green",
    "gold": "Gold",
    "green": "Green",
    "grey": "Grey",
    "gray": "Grey",
    "khaki": "Khaki",
    "lavender": "Lavender",
    "lime green": "Lime Green",
    "magenta": "Magenta",
    "maroon": "Maroon",
    "mauve": "Mauve",
    "multi": "Multi",
    "mushroom brown": "Mushroom Brown",
    "mustard": "Mustard",
    "navy": "Navy Blue",
    "navy blue": "Navy Blue",
    "nude": "Nude",
    "off white": "Off White",
    "olive": "Olive",
    "olive green": "Olive",
    "orange": "Orange",
    "peach": "Peach",
    "pink": "Pink",
    "purple": "Purple",
    "red": "Red",
    "rose": "Rose",
    "rust": "Rust",
    "sea green": "Sea Green",
    "silver": "Silver",
    "skin": "Skin",
    "steel": "Steel",
    "tan": "Tan",
    "taupe": "Taupe",
    "teal": "Teal",
    "turquoise": "Turquoise Blue",
    "turquoise blue": "Turquoise Blue",
    "white": "White",
    "yellow": "Yellow",
}

SCENE_PATTERNS = [
    r"\b(?:a|the)\s+(?:man|woman|boy|girl|person|model)\b",
    r"\b(?:standing|sitting|posing|walking)\b",
    r"\bin front of\b",
    r"\bbackground\b",
    r"\bwall\b",
    r"\broom\b",
    r"\bwearing\b",
]

FORBIDDEN_PREFIXES = [
    "description:",
    "caption:",
    "output:",
    "answer:",
    "generated description:",
    "product description:",
]

TRAILING_JUNK = [
    "###",
    "</s>",
    "<|endoftext|>",
    "[end]",
]

PATTERN_WORDS = [
    "checked",
    "check",
    "striped",
    "stripe",
    "solid",
    "printed",
    "print",
    "floral",
    "graphic",
    "embroidered",
    "polka",
    "plaid",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def title_item_type(article_type: Optional[str]) -> str:
    value = re.sub(r"\s+", " ", str(article_type or "").strip())
    if not value:
        return ""
    # Dataset labels are often plural; singular reads better in catalog copy.
    singular_map = {
        "Tshirts": "T-shirt",
        "Shirts": "shirt",
        "Kurtas": "kurta",
        "Tops": "top",
        "Sweatshirts": "sweatshirt",
        "Sweaters": "sweater",
        "Jackets": "jacket",
        "Kurtis": "kurti",
        "Tunics": "tunic",
        "Dupatta": "dupatta",
        "Jeans": "jeans",
        "Track Pants": "track pants",
    }
    return singular_map.get(value, value[:1].lower() + value[1:])


def normalize_colour_word(colour: Optional[str]) -> str:
    value = normalize(colour or "")
    return DATASET_COLOURS.get(value, str(colour or "").strip())


def find_colour(text: str) -> str:
    norm = normalize(text)
    for key in sorted(DATASET_COLOURS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", norm):
            return DATASET_COLOURS[key]
    return ""


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def strip_prompt_echo(text: str, prompt: Optional[str] = None) -> str:
    out = str(text or "").strip()
    prompts = [prompt or "", ECOMMERCE_PROTOCOL]
    for candidate in prompts:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        if normalize(out).startswith(normalize(candidate)):
            out = out[len(candidate) :].strip(" \n\t:.-")
        # Some models echo the tail of the instruction block before the answer.
        tail = "return only the description text"
        tail_index = normalize(out).find(tail)
        if tail_index >= 0:
            out = out[tail_index + len(tail) :].strip(" \n\t:.-")
    return out


def strip_repeated_prefixes(text: str) -> str:
    out = str(text or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in FORBIDDEN_PREFIXES:
            if out.lower().startswith(prefix):
                out = out[len(prefix) :].strip(" \n\t:.-")
                changed = True
    return out


def strip_trailing_junk(text: str) -> str:
    out = str(text or "").strip()
    for marker in TRAILING_JUNK:
        if marker in out:
            out = out.split(marker, 1)[0].strip()
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip(" \n\t-_:;")


def sentence_limit(text: str, max_sentences: int = 2) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    if not pieces:
        return ""
    return " ".join(pieces[:max_sentences])


def word_limit(text: str, max_words: int = 25) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words]).rstrip(" ,;:-")
    return clipped if clipped.endswith(".") else f"{clipped}."


def extract_pattern(text: str) -> str:
    norm = normalize(text)
    for word in PATTERN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", norm):
            if word in {"check", "checked", "plaid"}:
                return "checked"
            if word in {"stripe", "striped"}:
                return "striped"
            if word in {"print", "printed", "graphic", "floral"}:
                return word
            return word
    return ""


def product_focus_rewrite(
    text: str,
    article_type: Optional[str] = None,
    base_colour: Optional[str] = None,
) -> str:
    item_type = title_item_type(article_type)
    colour = normalize_colour_word(base_colour) or find_colour(text)
    pattern = extract_pattern(text)

    if not item_type and not colour:
        return text

    missing_type = bool(item_type and normalize(item_type) not in normalize(text))
    missing_colour = bool(colour and normalize(colour) not in normalize(text) and not find_colour(text))
    needs_rewrite = contains_any(text, SCENE_PATTERNS) or missing_type or missing_colour
    if not needs_rewrite:
        return text

    parts = []
    if colour:
        parts.append(colour.lower())
    if pattern and pattern not in {normalize(colour), normalize(item_type)}:
        parts.append(pattern)
    if item_type:
        parts.append(item_type)

    phrase = " ".join(parts).strip()
    if not phrase:
        return text
    return f"A {phrase}."


def ensure_token(text: str, token: str) -> str:
    if not token:
        return text
    if normalize(token) in normalize(text):
        return text
    if text:
        return f"{token.strip()} {text[:1].lower()}{text[1:]}"
    return token.strip()


def enforce_type_color(description: str, article_type: Optional[str], base_colour: Optional[str]) -> str:
    out = str(description or "").strip()
    item_type = title_item_type(article_type)
    colour = normalize_colour_word(base_colour)

    if item_type and normalize(item_type) not in normalize(out):
        out = ensure_token(out, item_type)
    has_colour = bool(find_colour(out))
    if colour and normalize(colour) not in normalize(out) and not has_colour:
        out = ensure_token(out, colour)
    return out


def clean_description(
    text: str,
    prompt: Optional[str] = None,
    article_type: Optional[str] = None,
    base_colour: Optional[str] = None,
    max_words: int = 25,
) -> str:
    out = strip_prompt_echo(text, prompt=prompt)
    out = strip_repeated_prefixes(out)
    out = strip_trailing_junk(out)
    out = sentence_limit(out, max_sentences=2)
    out = product_focus_rewrite(out, article_type=article_type, base_colour=base_colour)
    out = enforce_type_color(out, article_type=article_type, base_colour=base_colour)
    out = strip_trailing_junk(out)
    out = word_limit(out, max_words=max_words)
    return out
