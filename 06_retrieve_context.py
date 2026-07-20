# ============================================================
# 06_retrieve_context.py
# Turn retrieved chunks into a clean, bounded context block — but
# ONLY if the retrieved passages are actually relevant.
#
# Relevance gate: min-max normalized scores always produce a "best"
# match, so we judge relevance on ABSOLUTE raw scores:
#   - semantic (cosine) >= SEM_THRESHOLD  when embeddings are available
#   - lexical  (BM25)   >= BM25_THRESHOLD when only BM25 is available
# If nothing clears the bar, we return an empty context so the caller
# can fall back to a general-knowledge answer.
# ============================================================

from importlib import import_module

vectors = import_module("04_vector_representation")
hybrid_search = vectors.hybrid_search

TOP_K = 5
MAX_SOURCES = 4
WORD_BUDGET = 260
MIN_SCORE_RATIO = 0.30      # keep chunks within this ratio of the top hybrid score

SEM_THRESHOLD = 0.35        # min cosine similarity to count as "relevant"
BM25_THRESHOLD = 3.0        # min raw BM25 score when embeddings are unavailable


def _is_relevant(index, candidates):
    if not candidates:
        return False
    if index.semantic_ready:
        best_semantic = max(c["semantic_score"] for c in candidates)
        return best_semantic >= SEM_THRESHOLD
    best_lexical = max(c["lexical_score"] for c in candidates)
    return best_lexical >= BM25_THRESHOLD


def build_context(index, question, k=TOP_K, max_sources=MAX_SOURCES):
    candidates = hybrid_search(index, question, k=k * 2)

    # Absolute relevance gate: if nothing is truly related, treat as "not found".
    if not _is_relevant(index, candidates):
        return "", []

    max_score = max(c["score"] for c in candidates)
    selected, seen_docs, used_words = [], set(), 0

    for row in candidates:
        if max_score > 0 and row["score"] < max_score * MIN_SCORE_RATIO:
            continue
        if row["document_id"] in seen_docs:
            continue
        words = len(row["chunk_text"].split())
        if selected and used_words + words > WORD_BUDGET:
            continue
        selected.append(row)
        seen_docs.add(row["document_id"])
        used_words += words
        if len(selected) >= max_sources:
            break

    context = ""
    for i, row in enumerate(selected, start=1):
        context += f"[Source {i}] {row['doc_title']}\n{row['chunk_text']}\n\n"

    return context.strip(), selected
