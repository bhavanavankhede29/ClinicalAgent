"""Runtime configuration.

Values come from the environment, and from a local ``.env`` file at the project
root if one exists (copy ``.env.example`` to ``.env`` and fill it in). ``.env`` is
git-ignored. The Anthropic key is optional: without it the agent runs in
evidence-only mode (retrieval + provenance, no LLM synthesis).
"""
import os
from pathlib import Path

try:  # optional dependency; the app still runs without it
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

# Required only when ANTHROPIC_API_KEY is an "identity-linked" key (the newer type
# not bound to a single workspace). Find it in the Anthropic Console under the
# workspace's settings; it looks like "wrkspc_...". Leave blank for a classic
# workspace-scoped key.
ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()

# Seconds to wait on any single upstream medical data source. Kept short: sources
# are queried concurrently, so one slow endpoint should not stall the whole query.
REQUEST_TIMEOUT = float(os.environ.get("CLINICAL_AGENT_TIMEOUT", "12"))

# In-process TTL cache for upstream retrieval responses (seconds). Repeated terms
# (same drug across queries, retries within the LLM loop) then skip the network.
# Set to 0 to disable.
RETRIEVAL_CACHE_TTL = float(os.environ.get("CLINICAL_AGENT_CACHE_TTL", "900"))

# A contact string sent to NCBI per their E-utilities usage policy. Deliberately
# a non-personal placeholder; override with CLINICAL_AGENT_CONTACT if you want.
CONTACT = os.environ.get("CLINICAL_AGENT_CONTACT", "clinical-agent@localhost")

# Free subscription key for the NHS website Content API (api.nhs.uk). Optional —
# the NHS conditions source is skipped (with a note) when this is blank.
NHS_API_KEY = os.environ.get("NHS_API_KEY", "").strip()

# Public sandbox FHIR R4 server (synthetic patients only — never point this at a
# server holding real PHI). Used by /api/fhir/* to pull a demo patient's problem
# list, medications, and recent results as case context.
FHIR_BASE_URL = os.environ.get("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4").rstrip("/")

MAX_TOOL_ROUNDS = 6
MAX_TOKENS = 1600

# Console login. Two roles: 'admin' runs the console and does the final review;
# 'doctor' signs in at /doctor to review and approve generated treatment plans.
# Change the passwords for anything but local use, and set a fixed SESSION_SECRET
# so sessions survive a restart.
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DOCTOR_USER = os.environ.get("DOCTOR_USER", "doctor")
DOCTOR_PASSWORD = os.environ.get("DOCTOR_PASSWORD", "admindoc")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "") or os.urandom(24).hex()
SESSION_TTL = int(os.environ.get("SESSION_TTL", str(12 * 3600)))

# RAG / local knowledge base. Drop *.md / *.txt files in KNOWLEDGE_DIR, then
# rebuild the index with `python -m api.rag` or POST /api/rag/reindex.
_ROOT = Path(__file__).resolve().parent.parent

# Where generated treatment plans and their approval state are persisted (JSON).
REVIEW_STORE_PATH = Path(os.environ.get(
    "CLINICAL_AGENT_REVIEW_STORE", str(_ROOT / "data" / "treatment_plans.json")))

KNOWLEDGE_DIR = Path(os.environ.get("CLINICAL_AGENT_KNOWLEDGE_DIR", str(_ROOT / "knowledge")))
RAG_INDEX_PATH = KNOWLEDGE_DIR / ".kb_index.json"
RAG_TOP_K = int(os.environ.get("CLINICAL_AGENT_RAG_TOP_K", "4"))

# Chunking: 500-character windows with 100 characters of overlap to preserve
# context across boundaries.
RAG_CHUNK_CHARS = int(os.environ.get("CLINICAL_AGENT_RAG_CHUNK_CHARS", "500"))
RAG_CHUNK_OVERLAP = int(os.environ.get("CLINICAL_AGENT_RAG_CHUNK_OVERLAP", "100"))

# Vector embeddings. With VOYAGE_API_KEY set, chunks and queries are embedded with
# Voyage AI (Anthropic's recommended embedding provider) and retrieval is dense
# cosine similarity. Without it, retrieval falls back to a local TF-IDF vector.
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "").strip()
VOYAGE_BASE_URL = os.environ.get("VOYAGE_BASE_URL", "https://api.voyageai.com/v1").rstrip("/")
EMBEDDING_MODEL = os.environ.get("CLINICAL_AGENT_EMBED_MODEL", "voyage-3.5")
