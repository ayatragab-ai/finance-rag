# ============================================================
# 03_chunking.py
# Overlapping word-window chunker for financial passages.
#
# Short passages stay a single chunk; long filings are split with
# overlap so no figure or clause is cut at a boundary. search_text
# (title + passage) is what gets indexed; chunk_text (the passage)
# is what the model later reads.
# ============================================================

from importlib import import_module

preprocess_text = import_module("02_preprocessing").preprocess_text

CHUNK_SIZE = 180     # words per chunk (same as the notebook)
CHUNK_OVERLAP = 40   # overlapping words between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(documents):
    rows = []
    for document in documents:
        for i, piece in enumerate(chunk_text(document["text"])):
            rows.append(
                {
                    "chunk_id": f"{document['id']}_chunk{i}",
                    "document_id": document["id"],
                    "doc_title": document["doc_title"],
                    "is_current": document["is_current"],
                    "chunk_index": i,
                    "chunk_text": piece,
                    "search_text": preprocess_text(f"{document['doc_title']} {piece}"),
                }
            )
    return rows
