# Smart Inquiry Triage Assistant

Automated triage for customer inquiries: classifies an inquiry, retrieves similar
past cases, assigns a priority, routes it to a team, scores its own confidence,
and drafts a short internal note for the agent who picks it up.

Built with LangGraph, ChromaDB and the Google Gemini API.

---

## Setup

Requires Python 3.11 or 3.12. ChromaDB has no prebuilt wheel for 3.13.

```bash
git clone https://github.com/ankitghatole08/AI-Engineer-Case-Study.git
cd AI-Engineer-Case-Study

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a Google AI Studio key
(free, no card required, from https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=your_key_here
```

## Running

Build the vector store once. Takes about four minutes — the free embedding tier
is capped at 100 requests per minute and the script paces itself accordingly.

```bash
python -m src.ingest
```

Then start the app:

```bash
streamlit run app/app.py
```

Optional checks:

```bash
python scripts/smoke_test.py     # verify API connectivity and embedding quality
python -m src.graph              # run the pipeline on one inquiry
python -m scripts.evaluate       # hold-out evaluation
```

## Models

| Purpose | Model | Notes |
|---|---|---|
| Chat | `gemini-3.5-flash-lite` | Lightweight, free tier |
| Embeddings | `gemini-embedding-001` | 768 dimensions (default is 3072) |

Both are accessed through `langchain-google-genai`, so switching provider means
changing two class names in `src/config.py` and `src/nodes.py`.

---

## Architecture

Two halves. Ingestion runs once, offline. The pipeline runs per inquiry.

````mermaid
graph TD;
        __start__([start]):::first
        retrieve(retrieve)
        classify(classify)
        confidence(confidence)
        priority_and_route(priority_and_route)
        notes(notes)
        __end__([end]):::last
        __start__ --> retrieve;
        retrieve --> classify;
        classify --> confidence;
        confidence --> priority_and_route;
        priority_and_route --> notes;
        notes --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
````

| File | Role |
|---|---|
| `src/config.py` | Models, paths, defaults, category-to-queue map |
| `src/ingest.py` | Embeds past cases and taxonomy into ChromaDB |
| `src/retrieval.py` | Top-K similarity search |
| `src/nodes.py` | Triage logic — classification, confidence, priority, routing, notes |
| `src/graph.py` | LangGraph assembly |
| `src/main.py` | `triage_inquiry()`, the entry point the UI calls |
| `scripts/evaluate.py` | Hold-out evaluation |

Logic lives in `src/nodes.py` as plain functions; `src/graph.py` wraps each in a
thin node. That separation lets the evaluation script call the logic directly
without constructing graph state.

---

## Design decisions

### Two independent classifiers

Category is predicted twice, by methods that share no inputs.

**k-NN vote** — the retrieved cases carry human-assigned category labels. Summing
similarity per category and taking the winner is k-nearest-neighbours
classification, with ChromaDB as the index. Deterministic and free, since
retrieval already happened.

**LLM classification** — the inquiry and the eight taxonomy descriptions go to the
chat model, which returns one category name. It does **not** see the retrieved
cases; if it did it would tend to echo the k-NN answer, and their agreement
would carry no information.

They fail differently. k-NN fails when no similar case exists in the 300. The LLM
fails on boundary cases where two definitions both plausibly apply. That
independence is what makes their agreement meaningful.

The LLM's answer is the final category. The k-NN answer is a second opinion that
feeds the confidence score.

### Weighted rather than unweighted voting

Retrieved similarity scores cluster in a narrow band (typically 0.65–0.80), so
counting neighbours equally would let a marginal match count as much as a
near-exact one. Votes are weighted by similarity.

### Confidence

Three signals, weights summing to 1.0:

```
confidence = 0.5 × methods_agree      (1.0 or 0.0)
           + 0.3 × knn_agreement      (winner's share of total weight)
           + 0.2 × top_similarity     (nearest case's score)
```

Agreement carries the largest weight because it is the only signal drawn from two
independent sources — the other two both derive from the same retrieval.
`knn_agreement` measures how united the neighbours were. `top_similarity` catches
the case where both methods agree but nothing in the knowledge base is actually
close.

A consequence worth stating: a disagreement caps confidence at 0.5, so the two
classifiers splitting always escalates at the default threshold.

**Alternatives rejected.** Asking the model to self-report its confidence is one
line of code, but language models are poorly calibrated and it would make the
safety mechanism depend on an unreliable signal. Similarity alone says nothing
about whether the category is clear.

**Limitation.** The weights are chosen by reasoning, not fitted. With 300 cases
and no confidence labels there is nothing to fit against.

### Escalation flags, it does not block

A low-confidence inquiry still receives a category, priority, queue and notes.
The flag marks it for review. Blocking output would leave inquiries in limbo —
worse than the manual process being replaced. The UI also shows what each
classifier said, so a reviewer sees *why* the system was unsure.

### Routing is a lookup, not a model call

Category maps 1:1 to queue across all 300 past cases, so routing is a dictionary
lookup in `src/config.py`. Instant, free, and it cannot invent a team that does
not exist.

### Priority comes from the retrieved cases

As the brief specifies. The same weighted vote reads the `priority` column
instead of `category`. These labels are human judgements on real cases, so
inheriting them is more defensible than asking the model to guess.

### No chunking

Past cases are single-sentence inquiries far below the embedding token limit, and
each carries its own labels — splitting one would break the assumption that one
vector equals one labelled example. Taxonomy descriptions are around 100 words
and are deliberately kept as one vector each. Chunking would become necessary if
the knowledge base later included manuals or policy documents.

### Embedding dimensionality

`gemini-embedding-001` returns 3072 dimensions by default. This project uses 768.
The model is trained with Matryoshka Representation Learning, so the leading 768
values form a valid embedding on their own. For 300 short inquiries across 8
categories, 3072 is more resolution than the problem needs; 768 gives 4× less
storage and faster search. Reverting is a one-line change in `src/config.py`.

---

## Results

50 of the 300 cases were held back and the vector store was built from the other
250, so the test cases could not retrieve their own answer.

| Task | Guessing | This system |
|---|---|---|
| Category (8 types) | 19.6% | 100% |
| Priority (low/med/high) | 48.4% | 52.0% |

"Guessing" means always picking the most common answer, with no intelligence at
all. It is the bar worth measuring against — 52% on priority looks fine until you
see that guessing scores 48%.

**Category works.** Five times better than guessing, confirmed on a second run
with a different split. Read the 100% with caution though: the sample is small and
the provided inquiries are clean, one-sentence, and map neatly onto the category
definitions. Real customer messages have typos, several issues at once, and
missing context.

**Priority does not work.** Four points above guessing, consistently. Priority
depends on how urgent the wording is, while retrieval matches on what the inquiry
is about — two brake inquiries look similar whether one says "unsafe to drive" or
"whenever convenient". This is the first thing to fix.

**The confidence score is untested.** It exists to flag cases likely to be wrong,
and there were no wrong cases for it to catch.

### One case worth showing

*"The price in the configurator does not match what I was invoiced."*

Every retrieved past case was `configurator`; the language model read "invoiced"
and chose `billing`. Both are defensible — the inquiry spans both. Confidence
dropped to 0.46 and it was flagged for review. A single-classifier system would
have answered confidently and given no sign anything was ambiguous.

---


## Limitations and next steps

- **Priority prediction is near baseline.** The first thing to fix: either a
  dedicated urgency signal (keyword or sentiment based) or an LLM-based priority
  step that reads the inquiry rather than inheriting neighbour labels.
- **Confidence weights are reasoned, not fitted.** With labelled confidence data
  they could be tuned, or replaced with a small trained calibration model.
- **Evaluation set is small and clean.** Real inquiry data would be needed for a
  trustworthy accuracy figure.
- **A taxonomy embedding match is computed but unused.** `match_taxonomy()` in
  `src/retrieval.py` provides a third classification signal. It was left out to
  keep the combination rule simple and explainable.
- **Ingestion re-embeds on every run.** Incremental ingestion would only embed
  new cases.
- **Provider portability is untested.** The abstraction exists but no alternative
  provider has been wired up. A self-hosted model would be the route to keeping
  customer data in-house.