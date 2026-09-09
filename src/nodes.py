"""
Core triage logic, one function per pipeline step.

These are plain functions, not LangGraph nodes. src/graph.py wraps each one
in a thin node that reads and writes the shared state — keeping the logic
separate means it can be tested and evaluated without building a state dict.

    step 2  classify   -> knn_vote(), llm_classify()
    step 3  confidence -> (phase 7)
    step 4  priority   -> knn_vote(cases, "priority")
            route      -> (phase 8)
    step 5  notes      -> (phase 9)

Step 1, retrieval, lives in src/retrieval.py.
"""

import json
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI

from src import config

_llm = None
_taxonomy = None


# --- shared helpers ---

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=config.CHAT_MODEL)
    return _llm


def _get_taxonomy() -> list[dict]:
    global _taxonomy
    if _taxonomy is None:
        with open(config.TAXONOMY_JSON, encoding="utf-8") as f:
            _taxonomy = json.load(f)["categories"]
    return _taxonomy


def as_text(message) -> str:
    """Newer LangChain returns content as a list of typed blocks rather than
    a plain string.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


# --- step 2: classification, method A (deterministic) ---

def knn_vote(cases: list[dict], field: str = "category") -> tuple[str, float]:
    """Weighted vote over the retrieved cases.

    Weighting by similarity rather than counting equally matters because the
    scores sit in a narrow band; a near-exact match should outweigh a
    marginal one.

    Also used for priority in step 4 by passing field="priority".

    Returns the winning label and its share of the total weight (0.0-1.0).
    """
    if not cases:
        return "other", 0.0

    weights = defaultdict(float)
    for case in cases:
        weights[case[field]] += case["similarity"]

    total = sum(weights.values())
    winner = max(weights, key=weights.get)
    return winner, round(weights[winner] / total, 4)


# --- step 2: classification, method B (independent of method A) ---

def llm_classify(query: str) -> str:
    """Classify against the taxonomy definitions only.

    Deliberately does not see the retrieved cases: if it did, it would likely
    echo the k-NN answer and their agreement would carry no information.
    """
    categories = _get_taxonomy()
    valid = {c["name"] for c in categories}

    definitions = "\n".join(
        f"- {c['name']}: {c['description']}" for c in categories
    )

    prompt = f"""You are triaging a customer inquiry for an automotive company.

Categories:
{definitions}

Disambiguation rules:
- If the customer disputes coverage or cost of a repair, choose warranty, not service.
- If the problem is with software, connectivity or the app, choose technical, not service.
- If it concerns pricing or financing inside the car configurator, choose configurator, not billing.
- Use other only when no category above applies.

Inquiry: "{query}"

Reply with exactly one category name from the list. No punctuation, no explanation."""

    answer = as_text(_get_llm().invoke(prompt)).lower().strip().strip(".")

    # The model occasionally returns a near-miss; fall back rather than
    # letting an invalid category reach the routing step.
    return answer if answer in valid else "other"


# --- manual check: run both classifiers on one inquiry ---

if __name__ == "__main__":
    from src.retrieval import retrieve_past_cases

    demo = "My brakes are grinding and it feels unsafe to drive."
    cases = retrieve_past_cases(demo, top_k=5)

    knn_cat, agreement = knn_vote(cases, "category")
    llm_cat = llm_classify(demo)

    print(f"query:       {demo}")
    print(f"k-NN:        {knn_cat} (agreement {agreement})")
    print(f"LLM:         {llm_cat}")
    print(f"agree:       {knn_cat == llm_cat}")