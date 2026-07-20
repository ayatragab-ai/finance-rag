# ============================================================
# 05_create_chroma_store.py  (OPTIONAL - run once, not used at runtime)
#
# Persists the chunk embeddings into a local Chroma store. The live
# Streamlit app does NOT depend on Chroma at query time (it uses the
# in-memory hybrid search in 04), which keeps deployment reliable.
#
# Run locally with:   python 05_create_chroma_store.py
# Requires the optional deps in requirements-full.txt.
# ============================================================

import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from importlib import import_module
from pathlib import Path

build_index = import_module("04_vector_representation").build_index

DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "financeqa_docs"


def create_vector_store():
    import chromadb
    from chromadb.config import Settings

    index = build_index()
    if not index.semantic_ready:
        raise RuntimeError("Embeddings are not available; cannot build a Chroma store.")

    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.upsert(
        ids=[c["chunk_id"] for c in index.chunks],
        documents=[c["chunk_text"] for c in index.chunks],
        metadatas=[
            {
                "document_id": c["document_id"],
                "doc_title": c["doc_title"],
                "is_current": str(c["is_current"]),
            }
            for c in index.chunks
        ],
        embeddings=index.embeddings.tolist(),
    )
    return collection


if __name__ == "__main__":
    create_vector_store()
    print("Chroma vector store created at ./chroma_db")
