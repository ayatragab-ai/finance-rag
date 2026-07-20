# ============================================================
# 04_vector_representation.py
# Hybrid retrieval = BM25 (lexical) + dense embeddings (semantic).
#
#   hybrid = (1 - ALPHA) * BM25_norm + ALPHA * embedding_norm
#
# The embedding model loads gracefully: if sentence-transformers or
# the model download is unavailable, the system falls back to BM25
# only instead of crashing.
# ============================================================

from importlib import import_module

import numpy as np
from rank_bm25 import BM25Okapi

preprocessing = import_module("02_preprocessing")
load_documents = import_module("01_documents").load_documents
build_chunks = import_module("03_chunking").build_chunks

ALPHA = 0.5                       # weight of the semantic score
MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast sentence-transformer


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.size == 0 or scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


class VectorIndex:
    """Holds the documents, chunks, BM25 model, and (optionally) embeddings."""

    def __init__(self, max_docs=None):
        self.documents = load_documents() if max_docs is None else load_documents(max_docs)
        self.chunks = build_chunks(self.documents)
        self.tokenized = [
            preprocessing.simple_tokenize(c["search_text"]) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized)

        self.model = None
        self.embeddings = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME)
            self.embeddings = self.model.encode(
                [c["search_text"] for c in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[VectorIndex] Semantic backend unavailable, using BM25 only. ({exc})")

    @property
    def semantic_ready(self) -> bool:
        return self.model is not None and self.embeddings is not None


def build_index(max_docs=None) -> VectorIndex:
    return VectorIndex(max_docs=max_docs)


def hybrid_search(index: VectorIndex, query: str, k=5, alpha=ALPHA):
    clean_query = preprocessing.preprocess_text(query)
    tokens = preprocessing.simple_tokenize(clean_query)

    bm25_scores = np.array(index.bm25.get_scores(tokens))

    if index.semantic_ready:
        query_embedding = index.model.encode(
            [clean_query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        embedding_scores = index.embeddings @ query_embedding.reshape(-1)
        hybrid_scores = (
            (1 - alpha) * min_max_normalize(bm25_scores)
            + alpha * min_max_normalize(embedding_scores)
        )
    else:
        embedding_scores = np.zeros_like(bm25_scores)
        hybrid_scores = min_max_normalize(bm25_scores)

    ranking = np.argsort(hybrid_scores)[::-1][:k]
    return [
        {
            **index.chunks[i],
            "score": float(hybrid_scores[i]),
            "semantic_score": float(embedding_scores[i]),  # raw cosine (0..1)
            "lexical_score": float(bm25_scores[i]),         # raw BM25
        }
        for i in ranking
    ]
