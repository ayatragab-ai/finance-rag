# ============================================================
# 02_preprocessing.py
# Light, retrieval-safe normalization for FINANCIAL text.
#
# In finance the numbers and percentages ARE the answers, so cleaning
# must be gentle: we keep letters, digits, '.' and '-' and drop only
# punctuation and URLs. This mirrors the notebook's approach.
# ============================================================

import re


def normalize_lexical_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)   # keep digits, '.', '-'
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    return normalize_lexical_text(text)


def simple_tokenize(text: str):
    return re.findall(r"[a-z0-9\.\-]+", text.lower())
