"""
Smart Inquiry Triage Assistant — Streamlit chat interface.

This is the FRONTEND only. The actual triage logic (LLM, embeddings,
vector store, LangGraph workflow) is left for you (the candidate) to
implement in `triage_inquiry` below / in `src/main.py`.
"""

import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import triage_inquiry as run_triage

# --- Configuration (sidebar controls) ------------------------------------

st.set_page_config(page_title="Smart Inquiry Triage", page_icon="📨")
st.title("📨 Smart Inquiry Triage Assistant")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K past cases", min_value=1, max_value=10, value=3)
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


# --- Backend hook (IMPLEMENT ME) -----------------------------------------

def triage_inquiry(query: str, top_k: int, confidence_threshold: float) -> dict:
    """Delegates to the LangGraph pipeline in src/main.py."""
    return run_triage(query, top_k=top_k, confidence_threshold=confidence_threshold)


    # Example placeholder shape (remove once implemented):
    # return {
    #     "query": query,
    #     "category": "unknown",
    #     "priority": "low",
    #     "routed_queue": "general",
    #     "confidence": 0.0,
    #     "resolution_notes": "",
    #     "retrieved_past_cases": [],
    #     "escalated": True,
    # }


# --- Result rendering -----------------------------------------------------

def render_result(result: dict) -> None:
    """Render a triage result in the fixed output format."""
    st.markdown(f"**query:** {result.get('query', '')}")
    st.markdown(f"**category:** {result.get('category', '')}")
    st.markdown(f"**priority:** {result.get('priority', '')}")
    st.markdown(f"**routed queue:** {result.get('routed_queue', '')}")
    st.markdown(f"**confidence:** {result.get('confidence', '')}")
    st.markdown(f"**resolution notes:** {result.get('resolution_notes', '')}")

    past_cases = result.get("retrieved_past_cases", [])
    st.markdown("**retrieved past cases:**")
    if past_cases:
        for case in past_cases:
            if isinstance(case, dict):
                st.markdown(
                f"- `{case['similarity']:.3f}` "
                f"**[{case['category']} / {case['priority']}]** {case['text']}"
            )
            else:
                st.markdown(f"- {case}")
    else:
        st.markdown("- _none_")

    if result.get("escalated"):
        st.warning("⚠️ Escalated to human review (confidence below threshold).")

    # The two classifiers are independent, so showing both explains the score.
    if "knn_category" in result:
        agreement = "agree" if result.get("methods_agree") else "disagree"
        st.caption(
            f"k-NN vote: {result['knn_category']} · "
            f"LLM: {result['llm_category']} · methods {agreement}"
        )

# --- Chat / session view --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "result": dict|None}

# Replay history for this session.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        if turn["result"] is not None:
            render_result(turn["result"])
        else:
            st.error(turn.get("error", "No result."))

# New inquiry input.
query = st.chat_input("Paste a customer inquiry...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Triaging..."):
                result = triage_inquiry(query, top_k, confidence_threshold)
            render_result(result)
            st.session_state.history.append({"query": query, "result": result})
        except Exception as exc:
            msg = f"Triage failed: {exc}"
            st.error(msg)
            st.session_state.history.append(
                {"query": query, "result": None, "error": msg}
            )
