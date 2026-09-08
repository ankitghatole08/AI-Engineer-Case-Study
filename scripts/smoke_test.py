"""Phase 1 smoke test: verify Gemini chat + embeddings work."""

import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
assert os.environ.get("GOOGLE_API_KEY"), "GOOGLE_API_KEY not found in .env"

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"


def as_text(message) -> str:
    """Extract plain text from an AIMessage (content may be str or block list)."""
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


def cosine(a, b) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm


# --- 0. What models does this key actually have access to? ---
print("--- available models ---")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for m in client.models.list():
    name = m.name.replace("models/", "")
    if "flash" in name or "embedding" in name:
        print(" ", name)

# --- 1. Chat model ---
# Note: gemini-3.5-flash-lite uses fixed sampling defaults and ignores temperature.
print("\n--- chat test ---")
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
print(" response:", as_text(llm.invoke("Reply with exactly one word: OK")))

# --- 2. Embedding model ---
print("\n--- embedding test ---")
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    output_dimensionality=768,
)

doc_vec = embeddings.embed_documents(
    ["My brakes are grinding and the car feels unsafe."],
    task_type="RETRIEVAL_DOCUMENT",
)[0]

query_vec = embeddings.embed_query(
    "grinding noise when braking",
    task_type="RETRIEVAL_QUERY",
)

print(" document vector length:", len(doc_vec))
print(" query vector length:   ", len(query_vec))

# --- 3. Do related sentences actually land close together? ---
print("\n--- similarity checks ---")
print(" related pair:  ", round(cosine(doc_vec, query_vec), 4))

unrelated_vec = embeddings.embed_query(
    "How do I unsubscribe from the newsletter?",
    task_type="RETRIEVAL_QUERY",
)
print(" unrelated pair:", round(cosine(doc_vec, unrelated_vec), 4))