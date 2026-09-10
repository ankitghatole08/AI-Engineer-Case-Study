"""
Hold-out evaluation of the classification and confidence design.

Splits the 300 labelled cases into a knowledge base and a test set, builds a
temporary Chroma collection from the knowledge base only, then predicts the
held-out cases. Because the test cases are absent from the store, retrieval
cannot simply find and copy their own labels.

Measures:
    - k-NN accuracy alone
    - LLM accuracy alone
    - combined system accuracy (the LLM's answer, as used in production)
    - accuracy split by whether confidence cleared the threshold

Usage:
    python -m scripts.evaluate
"""

import chromadb
import pandas as pd
from sklearn.metrics import classification_report

from src import config, nodes
from src.ingest import embed_in_batches, get_embeddings

TEST_SIZE = 50
RANDOM_SEED = 7
EVAL_COLLECTION = "eval_knowledge_base"


def split_cases():
    df = pd.read_csv(config.PAST_CASES_CSV)
    test = df.sample(n=TEST_SIZE, random_state=RANDOM_SEED)
    knowledge = df.drop(test.index)
    return knowledge, test


def build_eval_collection(client, embeddings, knowledge: pd.DataFrame):
    """Separate collection so the production store is left untouched."""
    try:
        client.delete_collection(EVAL_COLLECTION)
    except Exception:
        pass

    texts = knowledge["inquiry_text"].tolist()
    vectors = embed_in_batches(embeddings, texts)

    collection = client.get_or_create_collection(
        name=EVAL_COLLECTION,
        metadata={"hnsw:space": config.DISTANCE_METRIC},
    )
    collection.add(
        ids=knowledge["case_id"].tolist(),
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "case_id": row.case_id,
                "category": row.category,
                "priority": row.priority,
                "routed_queue": row.routed_queue,
            }
            for row in knowledge.itertuples()
        ],
    )
    return collection


def retrieve_from(collection, embeddings, query: str, top_k: int) -> list[dict]:
    vector = embeddings.embed_query(query, task_type="RETRIEVAL_QUERY")
    result = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "case_id": meta["case_id"],
            "text": doc,
            "category": meta["category"],
            "priority": meta["priority"],
            "routed_queue": meta["routed_queue"],
            "similarity": round(1.0 - dist, 4),
        }
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def main() -> None:
    knowledge, test = split_cases()
    print(f"knowledge base: {len(knowledge)} cases")
    print(f"test set:       {len(test)} cases\n")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embeddings = get_embeddings()

    print("--- building evaluation knowledge base ---")
    collection = build_eval_collection(client, embeddings, knowledge)

    print("\n--- predicting held-out cases ---")
    rows = []
    for i, case in enumerate(test.itertuples(), start=1):
        cases = retrieve_from(
            collection, embeddings, case.inquiry_text, config.DEFAULT_TOP_K
        )
        knn_cat, agreement = nodes.knn_vote(cases, "category")
        llm_cat = nodes.llm_classify(case.inquiry_text)
        scored = nodes.score_confidence(knn_cat, llm_cat, agreement, cases)
        knn_priority, _ = nodes.determine_priority(cases)

        rows.append({
            "true_category": case.category,
            "knn_category": knn_cat,
            "llm_category": llm_cat,
            "true_priority": case.priority,
            "pred_priority": knn_priority,
            "confidence": scored["confidence"],
            "methods_agree": scored["methods_agree"],
        })
        print(f"  {i}/{len(test)}", end="\r")

    results = pd.DataFrame(rows)

    # --- accuracy of each method ---
    knn_acc = (results.knn_category == results.true_category).mean()
    llm_acc = (results.llm_category == results.true_category).mean()
    priority_acc = (results.pred_priority == results.true_priority).mean()

    # Majority-class baselines: what you get with no intelligence at all.
    cat_baseline = knowledge.category.value_counts(normalize=True).max()
    pri_baseline = knowledge.priority.value_counts(normalize=True).max()

    print("\n\n--- category accuracy ---")
    print(f"  majority-class baseline  {cat_baseline:.1%}")
    print(f"  k-NN alone               {knn_acc:.1%}")
    print(f"  LLM alone (system)       {llm_acc:.1%}")

    print("\n--- priority accuracy ---")
    print(f"  majority-class baseline  {pri_baseline:.1%}")
    print(f"  k-NN vote                {priority_acc:.1%}")

    # --- does the confidence score identify its own mistakes? ---
    threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
    high = results[results.confidence >= threshold]
    low = results[results.confidence < threshold]

    print(f"\n--- confidence check (threshold {threshold}) ---")
    if len(high):
        acc = (high.llm_category == high.true_category).mean()
        print(f"  above threshold  {len(high):>2} cases, {acc:.1%} correct")
    if len(low):
        acc = (low.llm_category == low.true_category).mean()
        print(f"  below threshold  {len(low):>2} cases, {acc:.1%} correct")
    if not len(low):
        print("  no cases fell below threshold")

    print("\n--- per-category detail (system prediction) ---")
    print(classification_report(
        results.true_category, results.llm_category, zero_division=0
    ))

    client.delete_collection(EVAL_COLLECTION)
    results.to_csv("evaluation_results.csv", index=False)
    print("Per-case results written to evaluation_results.csv")


if __name__ == "__main__":
    main()