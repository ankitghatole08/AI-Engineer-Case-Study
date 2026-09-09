"""
Entry point for the triage system.

app/app.py imports triage_inquiry() from here. The compiled graph is built
once at import time and reused — Streamlit re-runs its script on every
interaction, so rebuilding per call would add seconds to every query.
"""

from src import config
from src.graph import build_graph

_app = build_graph()


def triage_inquiry(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Run one inquiry through the triage pipeline.

    Returns the seven fields required by the case study, plus the internal
    signals behind the confidence score so the UI can show its reasoning.
    """
    state = _app.invoke({
        "query": query,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
    })

    return {
        "query": query,
        "category": state["category"],
        "priority": state["priority"],
        "routed_queue": state["routed_queue"],
        "confidence": state["confidence"],
        "resolution_notes": state["resolution_notes"],
        "retrieved_past_cases": state["retrieved_cases"],
        # Supporting signals — not required by the brief, but they make the
        # confidence score explainable in the UI.
        "escalated": state["escalated"],
        "knn_category": state["knn_category"],
        "llm_category": state["llm_category"],
        "methods_agree": state["methods_agree"],
    }


if __name__ == "__main__":
    result = triage_inquiry("My brakes are grinding and it feels unsafe to drive.")
    for key, value in result.items():
        if key != "retrieved_past_cases":
            print(f"{key:22} {value}")