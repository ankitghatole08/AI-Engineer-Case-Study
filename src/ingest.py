"""
Embeds the past cases and the taxonomy into ChromaDB.

Runs once before the app starts, producing two collections in ./chroma_db:
    past_cases  - 300 historical inquiries with category/priority/queue labels
    taxonomy    - 8 category definitions, used for classification

Usage:
    python -m src.ingest
"""

import json
import re
import time

import chromadb
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:

    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        output_dimensionality=config.EMBED_DIM,
    )


def embed_in_batches(embeddings, texts, batch_size=90, pause=62.0, max_retries=5):
    """The free tier counts each text as one request, capped at 100/minute,
    so batches are paced rather than sent back-to-back. Retries honour the
    delay the API reports instead of failing the whole run.
    """
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        print(f"  embedding {start + 1}-{start + len(batch)} of {len(texts)}...")

        for attempt in range(max_retries):
            try:
                vectors.extend(
                    embeddings.embed_documents(batch, task_type="RETRIEVAL_DOCUMENT")
                )
                break
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == max_retries - 1:
                    raise
                match = re.search(r"retry in ([\d.]+)s", str(exc))
                wait = float(match.group(1)) + 5 if match else 65.0
                print(f"    rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)

        if start + batch_size < len(texts):
            print(f"    pausing {pause:.0f}s to respect the rate limit...")
            time.sleep(pause)

    return vectors


def ingest_past_cases(client, embeddings) -> int:
    """Embed the historical inquiries, keeping their labels as metadata."""
    df = pd.read_csv(config.PAST_CASES_CSV)
    print(f"Loaded {len(df)} past cases.")

    texts = df["inquiry_text"].tolist()
    vectors = embed_in_batches(embeddings, texts)

    # hnsw:space overrides Chroma's L2 default; cosine suits text embeddings
    # and can only be set at collection creation.
    collection = client.get_or_create_collection(
        name=config.CASES_COLLECTION,
        metadata={"hnsw:space": config.DISTANCE_METRIC},
    )
    collection.add(
        ids=df["case_id"].tolist(),
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "case_id": row.case_id,
                "category": row.category,
                "priority": row.priority,
                "routed_queue": row.routed_queue,
            }
            for row in df.itertuples()
        ],
    )
    return collection.count()


def ingest_taxonomy(client, embeddings) -> int:
    """Embed one vector per category definition.

    Allows classification without an LLM call: embed the inquiry and see
    which category description it sits closest to.
    """
    with open(config.TAXONOMY_JSON, encoding="utf-8") as f:
        categories = json.load(f)["categories"]
    print(f"Loaded {len(categories)} taxonomy categories.")

    # Keywords are included because the descriptions alone under-represent
    # the concrete vocabulary customers actually use.
    texts = [
        f"{c['name']}: {c['description']} Keywords: {', '.join(c['keywords'])}"
        for c in categories
    ]
    vectors = embed_in_batches(embeddings, texts)

    collection = client.get_or_create_collection(
        name=config.TAXONOMY_COLLECTION,
        metadata={"hnsw:space": config.DISTANCE_METRIC},
    )
    collection.add(
        ids=[c["name"] for c in categories],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"category": c["name"]} for c in categories],
    )
    return collection.count()


def main() -> None:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Drop existing collections so re-running replaces rather than duplicates
    # vectors, which would otherwise skew the nearest-neighbour vote.
    for name in (config.CASES_COLLECTION, config.TAXONOMY_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    embeddings = get_embeddings()

    print("\n--- ingesting past cases ---")
    n_cases = ingest_past_cases(client, embeddings)

    print("\n--- ingesting taxonomy ---")
    n_tax = ingest_taxonomy(client, embeddings)

    print(f"\nDone. {n_cases} past cases and {n_tax} categories stored in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()