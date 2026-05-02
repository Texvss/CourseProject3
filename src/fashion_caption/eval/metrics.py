import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu


def ensure_nltk():
    nltk.download("punkt", quiet=True)


def bleu1(df: pd.DataFrame, ref_col: str = "reference", gen_col: str = "generated") -> float:
    scores = []
    for _, r in df.iterrows():
        ref_tok = nltk.word_tokenize(str(r[ref_col]).lower())
        gen_tok = nltk.word_tokenize(str(r[gen_col]).lower())
        if len(ref_tok) == 0 or len(gen_tok) == 0:
            continue
        scores.append(sentence_bleu([ref_tok], gen_tok, weights=(1, 0, 0, 0)))
    return float(np.mean(scores)) if scores else 0.0
