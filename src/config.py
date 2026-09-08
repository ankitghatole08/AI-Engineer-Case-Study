from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAST_CASES_CSV = DATA_DIR / "past_cases.csv"
TAXONOMY_JSON = DATA_DIR / "taxonomy.json"
CHROMA_DIR = ROOT / "chroma_db"

# --- Models ---
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

# --- Vector store ---
CASES_COLLECTION = "past_cases"
TAXONOMY_COLLECTION = "taxonomy"
DISTANCE_METRIC = "cosine"

# --- Triage defaults ---
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# --- Routing: category -> queue (1:1 mapping observed in past_cases.csv) ---
QUEUE_MAP = {
    "service": "Service Scheduling Team",
    "technical": "Technical Support Team",
    "ordering": "Order Management Team",
    "billing": "Billing & Payments Team",
    "configurator": "Digital Product / Configurator Support",
    "warranty": "Warranty Claims Team",
    "general": "Customer Care Team",
    "other": "General Correspondence / Triage Queue",
}

FALLBACK_QUEUE = "General Correspondence / Triage Queue"