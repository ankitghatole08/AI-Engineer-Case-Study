# AI-Engineer-Case-Study
AI-powered Smart Inquiry Triage Assistant

This repository is the starting point for the **AI Engineer case study**.

Your task is to build a prototype that takes a customer inquiry and
automatically **classifies** it, assigns a **priority**, **routes** it to the
right queue, and drafts short **resolution notes** — using similar past cases
as context.

Please read the full task description in
[case_study_smart_inquiry_triage.pdf](case_study_smart_inquiry_triage.pdf)
before you start.

## What's included

```
app/app.py           # Streamlit chat UI (provided) — calls triage_inquiry()
src/main.py          # Implement your triage pipeline here (backend)
data/past_cases.csv  # Historical inquiries: category, priority, routed_queue
data/taxonomy.json   # Canonical categories: name, description, keywords
requirements.txt     # Dependencies
```

The **Streamlit frontend is already provided**. Feel free to modify the frontend.
You have to implement the backend
function `triage_inquiry(query, top_k, confidence_threshold)` in `app/app.py`
(you may delegate to `src/main.py`).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open http://localhost:8501. Until `triage_inquiry` is implemented, the app
shows a "Backend not implemented yet" message.
