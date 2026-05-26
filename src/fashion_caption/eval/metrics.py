import math
import re
from collections import Counter
from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from fashion_caption.postprocess.text import normalize, normalize_colour_word, title_item_type
from fashion_caption.prompts import ECOMMERCE_PROTOCOL


FORBIDDEN_PATTERNS = [
    r"\b(man|woman|boy|girl|person|people|model|wearing|posing|standing|sitting)\b",
    r"\b(background|wall|room|floor|street|outdoor|indoor)\b",
    r"\b(brand|price|sale|discount|cheap|premium)\b",
    r"\b(cotton|polyester|silk|wool|leather|denim|material|fabric)\b",
]

PROMPT_ECHO_PATTERNS = [
    r"\byou are generating\b",
    r"\brules:\b",
    r"\breturn only\b",
    r"\bknown product type\b",
    r"\bknown main colou?r\b",
]

RELAXED_TYPE_ALIASES = {
    "Tshirts": ["tshirt", "tshirts", "t-shirt", "t-shirts", "tee", "tees", "shirt", "shirts"],
    "Shirts": ["shirt", "shirts", "button-down", "button down"],
    "Kurtas": ["kurta", "kurtas", "kurti", "kurtis"],
    "Tops": ["top", "tops", "tank", "tank top", "blouse", "tee", "t-shirt", "shirt"],
    "Sweatshirts": ["sweatshirt", "sweatshirts", "hoodie", "hoodies"],
    "Sweaters": ["sweater", "sweaters", "pullover", "pullovers", "jumper", "jumpers"],
    "Jackets": ["jacket", "jackets", "coat", "coats"],
    "Kurtis": ["kurti", "kurtis", "kurta", "kurtas"],
    "Tunics": ["tunic", "tunics"],
    "Dupatta": ["dupatta", "dupattas", "scarf", "shawl"],
}


def ensure_nltk():
    """Compatibility hook kept for old callers; metrics use regex tokenization."""
    return None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", str(text or "").lower())


def bleu1(df: pd.DataFrame, ref_col: str = "reference", gen_col: str = "generated") -> float:
    scores = []
    for _, row in df.iterrows():
        ref_tok = tokenize(str(row.get(ref_col, "")))
        gen_tok = tokenize(str(row.get(gen_col, "")))
        if not ref_tok or not gen_tok:
            continue
        overlap = sum((Counter(ref_tok) & Counter(gen_tok)).values())
        precision = overlap / len(gen_tok)
        brevity_penalty = 1.0 if len(gen_tok) > len(ref_tok) else math.exp(1 - len(ref_tok) / len(gen_tok))
        scores.append(brevity_penalty * precision)
    return float(np.mean(scores)) if scores else 0.0


def wilson_interval(p_hat: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p_hat = min(max(float(p_hat), 0.0), 1.0)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def keyword_present(text: str, keyword: str) -> bool:
    keyword = normalize(keyword)
    if not keyword:
        return False
    return keyword in normalize(text)


def relaxed_type_matcher(text: str, article_type: Optional[str]) -> bool:
    terms = RELAXED_TYPE_ALIASES.get(str(article_type or "").strip())
    if not terms:
        item_type = title_item_type(article_type)
        terms = [item_type] if item_type else []
    norm = normalize(text)
    return any(re.search(rf"\b{re.escape(normalize(term))}\b", norm) for term in terms if normalize(term))


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    pieces = [piece for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if piece.strip()]
    if pieces:
        return len(pieces)
    return 1 if str(text or "").strip() else 0


def protocol_compliance(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if any(marker in value for marker in ("```", "\n", "\t", "* ", "- ")):
        return False
    return 1 <= sentence_count(value) <= 2 and word_count(value) <= 25


def forbidden_rate(text: str) -> bool:
    norm = normalize(text)
    return any(re.search(pattern, norm) for pattern in FORBIDDEN_PATTERNS)


def prompt_echo_rate(text: str, prompt: Optional[str] = None) -> bool:
    norm = normalize(text)
    prompts = [prompt or "", ECOMMERCE_PROTOCOL]
    if any(candidate and norm.startswith(normalize(candidate)[:40]) for candidate in prompts):
        return True
    return any(re.search(pattern, norm) for pattern in PROMPT_ECHO_PATTERNS)


def _rate(values: Iterable[bool]) -> tuple[float, int, int, float, float]:
    items = list(values)
    n = len(items)
    successes = int(sum(bool(item) for item in items))
    rate = successes / n if n else 0.0
    low, high = wilson_interval(rate, n)
    return rate, successes, n, low, high


def attribute_accuracy(
    df: pd.DataFrame,
    gen_col: str = "generated_text",
    type_col: str = "articleType",
    color_col: str = "baseColour",
) -> dict:
    metrics = attribute_metrics(df, gen_col=gen_col, type_col=type_col, color_col=color_col)
    return {
        "type_keyword_accuracy": metrics["type_keyword_accuracy"],
        "color_keyword_accuracy": metrics["color_keyword_accuracy"],
    }


def attribute_metrics(
    df: pd.DataFrame,
    gen_col: str = "generated_text",
    type_col: str = "articleType",
    color_col: str = "baseColour",
) -> dict:
    type_scores = []
    relaxed_type_scores = []
    color_scores = []
    for _, row in df.iterrows():
        generated = str(row.get(gen_col, ""))
        article_type = title_item_type(row.get(type_col, ""))
        raw_article_type = row.get(type_col, "")
        base_colour = str(row.get(color_col, ""))
        if article_type:
            type_scores.append(keyword_present(generated, article_type))
            relaxed_type_scores.append(relaxed_type_matcher(generated, raw_article_type))
        if base_colour:
            expected_colour = normalize_colour_word(base_colour)
            color_scores.append(keyword_present(generated, expected_colour))

    type_rate, type_success, type_total, type_low, type_high = _rate(type_scores)
    relaxed_rate, relaxed_success, relaxed_total, relaxed_low, relaxed_high = _rate(relaxed_type_scores)
    color_rate, color_success, color_total, color_low, color_high = _rate(color_scores)
    return {
        "type_keyword_accuracy": type_rate,
        "type_keyword_successes": type_success,
        "type_keyword_total": type_total,
        "type_keyword_ci_low": type_low,
        "type_keyword_ci_high": type_high,
        "type_keyword_relaxed_accuracy": relaxed_rate,
        "type_keyword_relaxed_successes": relaxed_success,
        "type_keyword_relaxed_total": relaxed_total,
        "type_keyword_relaxed_ci_low": relaxed_low,
        "type_keyword_relaxed_ci_high": relaxed_high,
        "color_keyword_accuracy": color_rate,
        "color_keyword_successes": color_success,
        "color_keyword_total": color_total,
        "color_keyword_ci_low": color_low,
        "color_keyword_ci_high": color_high,
    }


def protocol_metrics(df: pd.DataFrame, gen_col: str = "generated_text", prompt_col: str = "prompt") -> dict:
    compliance_scores = []
    forbidden_scores = []
    echo_scores = []
    for _, row in df.iterrows():
        text = str(row.get(gen_col, ""))
        prompt = str(row.get(prompt_col, ""))
        compliance_scores.append(protocol_compliance(text))
        forbidden_scores.append(forbidden_rate(text))
        echo_scores.append(prompt_echo_rate(text, prompt=prompt))

    compliance_rate, compliance_success, compliance_total, compliance_low, compliance_high = _rate(compliance_scores)
    forbidden_mentions_rate, forbidden_success, _, forbidden_low, forbidden_high = _rate(forbidden_scores)
    echo_rate, echo_success, _, echo_low, echo_high = _rate(echo_scores)
    return {
        "protocol_compliance_rate": compliance_rate,
        "protocol_compliance_successes": compliance_success,
        "protocol_compliance_total": compliance_total,
        "protocol_compliance_ci_low": compliance_low,
        "protocol_compliance_ci_high": compliance_high,
        "forbidden_mentions_rate": forbidden_mentions_rate,
        "forbidden_mentions_successes": forbidden_success,
        "forbidden_mentions_ci_low": forbidden_low,
        "forbidden_mentions_ci_high": forbidden_high,
        "prompt_echo_rate": echo_rate,
        "prompt_echo_successes": echo_success,
        "prompt_echo_ci_low": echo_low,
        "prompt_echo_ci_high": echo_high,
    }
