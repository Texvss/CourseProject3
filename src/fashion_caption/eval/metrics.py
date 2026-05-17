import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

from fashion_caption.postprocess.text import normalize, normalize_colour_word, title_item_type


def ensure_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def bleu1(df: pd.DataFrame, ref_col: str = "reference", gen_col: str = "generated") -> float:
    ensure_nltk()
    scores = []
    for _, r in df.iterrows():
        ref_tok = nltk.word_tokenize(str(r[ref_col]).lower())
        gen_tok = nltk.word_tokenize(str(r[gen_col]).lower())
        if len(ref_tok) == 0 or len(gen_tok) == 0:
            continue
        scores.append(sentence_bleu([ref_tok], gen_tok, weights=(1, 0, 0, 0)))
    return float(np.mean(scores)) if scores else 0.0


def keyword_present(text: str, keyword: str) -> bool:
    keyword = normalize(keyword)
    if not keyword:
        return False
    return keyword in normalize(text)


def attribute_accuracy(
    df: pd.DataFrame,
    gen_col: str = "generated_text",
    type_col: str = "articleType",
    color_col: str = "baseColour",
) -> dict:
    type_scores = []
    color_scores = []
    for _, row in df.iterrows():
        generated = str(row.get(gen_col, ""))
        article_type = title_item_type(row.get(type_col, ""))
        base_colour = str(row.get(color_col, ""))
        if article_type:
            type_scores.append(keyword_present(generated, article_type))
        if base_colour:
            expected_colour = normalize_colour_word(base_colour)
            color_scores.append(keyword_present(generated, expected_colour))
    return {
        "type_keyword_accuracy": float(np.mean(type_scores)) if type_scores else 0.0,
        "color_keyword_accuracy": float(np.mean(color_scores)) if color_scores else 0.0,
    }
