"""
Query-time retrieval against the ChromaDB store built by src/ingest.py.

Returns the K most similar past cases for an inquiry, each with its
category, priority, routed queue and similarity score.
"""

import chromadb

from src import config
from src.ingest import get_embeddings

_client = None
_embeddings = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def embed_query(text: str) -> list[float]:
    """RETRIEVAL_QUERY is the counterpart to the RETRIEVAL_DOCUMENT used at
    ingestion; the pair is what makes short questions land near the longer
    cases that answer them.
    """
    return _get_embeddings().embed_query(text, task_type="RETRIEVAL_QUERY")


def retrieve_past_cases(query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    collection = _get_client().get_collection(config.CASES_COLLECTION)
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    cases = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        cases.append({
            "case_id": meta["case_id"],
            "text": doc,
            "category": meta["category"],
            "priority": meta["priority"],
            "routed_queue": meta["routed_queue"],
            # Chroma returns cosine distance; invert it so higher means closer.
            "similarity": round(1.0 - dist, 4),
        })
    return cases


def match_taxonomy(query: str, top_k: int = 3) -> list[dict]:
    """Nearest category definitions — a classification signal that needs no
    LLM call.
    """
    collection = _get_client().get_collection(config.TAXONOMY_COLLECTION)
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    return [
        {"category": meta["category"], "similarity": round(1.0 - dist, 4)}
        for meta, dist in zip(result["metadatas"][0], result["distances"][0])
    ]


if __name__ == "__main__":
    demo = "My brakes are grinding and it feels unsafe to drive."
    print(f"query: {demo}\n")

    print("--- nearest past cases ---")
    for case in retrieve_past_cases(demo, top_k=5):
        print(f"  {case['similarity']:.3f}  [{case['category']}/{case['priority']}]  {case['text']}")

    print("\n--- nearest categories ---")
    for match in match_taxonomy(demo):
        print(f"  {match['similarity']:.3f}  {match['category']}")