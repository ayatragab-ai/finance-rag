# ============================================================
# 01_documents.py
# Load the financial corpus (FinanceQA-style / 10-K filings) from a
# LOCAL JSON file shipped with the project — no internet needed.
#
# Each entry has the same shape as the notebook's dataset:
#   question, answer, context (the answer-bearing passage), ticker.
# The app searches over the context passages.
#
# To use more data, add entries to financeqa_data.json (same format).
# ============================================================

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "financeqa_data.json"


def load_documents(max_docs: int = None):
    """Return a list of {id, doc_title, is_current, question, answer, text}."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    documents, seen = [], set()
    for i, row in enumerate(rows):
        context = str(row.get("context", "")).strip()
        if not context:
            continue
        key = context.lower()
        if key in seen:                 # de-duplicate identical passages
            continue
        seen.add(key)

        documents.append(
            {
                "id": f"doc_{i}",
                "doc_title": str(row.get("ticker", "FinanceQA")).strip()[:60] or "FinanceQA",
                "is_current": True,
                "question": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
                "text": context,
            }
        )
        if max_docs is not None and len(documents) >= max_docs:
            break

    return documents
