"""
LangGraph assembly. Nodes are thin wrappers over the logic in src/nodes.py —
each reads what it needs from the shared state and returns only the fields
it produces. LangGraph merges those updates and passes state to the next node.

Flow:
    START -> retrieve -> classify -> confidence -> priority_and_route -> notes -> END

Escalation is a field on the state, not a branch. A low-confidence inquiry
still gets a priority, a queue and notes; the flag only marks it for review.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import config, nodes
from src.retrieval import retrieve_past_cases


class TriageState(TypedDict, total=False):
    # inputs
    query: str
    top_k: int
    confidence_threshold: float

    # step 1
    retrieved_cases: list[dict]

    # step 2
    knn_category: str
    knn_agreement: float
    llm_category: str
    category: str

    # step 3
    confidence: float
    methods_agree: bool
    top_similarity: float
    escalated: bool

    # step 4
    priority: str
    priority_agreement: float
    routed_queue: str

    # step 5
    resolution_notes: str

#retrieve node
def retrieve_node(state: TriageState) -> dict:
    cases = retrieve_past_cases(
        state["query"],
        top_k=state.get("top_k", config.DEFAULT_TOP_K),
    )
    return {"retrieved_cases": cases}

#classify node
def classify_node(state: TriageState) -> dict:
    cases = state["retrieved_cases"]
    knn_category, knn_agreement = nodes.knn_vote(cases, "category")
    llm_category = nodes.llm_classify(state["query"])
    return {
        "knn_category": knn_category,
        "knn_agreement": knn_agreement,
        "llm_category": llm_category,
        # The LLM's answer is the final category; k-NN is the second opinion
        # that feeds confidence.
        "category": llm_category,
    }

#confidence node
def confidence_node(state: TriageState) -> dict:
    scored = nodes.score_confidence(
        state["knn_category"],
        state["llm_category"],
        state["knn_agreement"],
        state["retrieved_cases"],
    )
    threshold = state.get("confidence_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD)
    return {
        **scored,
        "escalated": nodes.should_escalate(scored["confidence"], threshold),
    }

#priority and route node
def priority_and_route_node(state: TriageState) -> dict:
    priority, priority_agreement = nodes.determine_priority(state["retrieved_cases"])
    return {
        "priority": priority,
        "priority_agreement": priority_agreement,
        "routed_queue": nodes.route_to_queue(state["category"]),
    }

#notes node
def notes_node(state: TriageState) -> dict:
    return {
        "resolution_notes": nodes.generate_resolution_notes(
            state["query"],
            state["category"],
            state["priority"],
            state["retrieved_cases"],
        )
    }


#LangGraph assembly
def build_graph():
    graph = StateGraph(TriageState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("classify", classify_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("priority_and_route", priority_and_route_node)
    graph.add_node("notes", notes_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "confidence")
    graph.add_edge("confidence", "priority_and_route")
    graph.add_edge("priority_and_route", "notes")
    graph.add_edge("notes", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "query": "My brakes are grinding and it feels unsafe to drive.",
        "top_k": 5,
        "confidence_threshold": 0.5,
    })

    for key in ("category", "priority", "routed_queue", "confidence",
                "escalated", "resolution_notes"):
        print(f"{key:18} {result[key]}")