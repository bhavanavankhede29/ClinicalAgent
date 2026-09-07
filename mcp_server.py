"""Clinical Agent — MCP server.

Exposes the same evidence-retrieval layer the web app uses as Model Context
Protocol tools over stdio. No API keys required for any source.

Sources: a local knowledge base (RAG — TF-IDF over files in knowledge/), PubMed
(NCBI E-utilities), ClinicalTrials.gov (API v2), openFDA (drug labels + FAERS),
RxNorm (RxNav), and four patient-facing sources used in place of WebMD (no API):
MedlinePlus health topics, MedlinePlus Connect (by drug code), MyHealthfinder
(health.gov), and the NHS website Content API (needs NHS_API_KEY).

Run directly:      python mcp_server.py
Register (Claude Code):
    claude mcp add clinical-agent -- \
        "c:/Users/gradb/ClinicalAgent/.venv/Scripts/python.exe" \
        "c:/Users/gradb/ClinicalAgent/mcp_server.py"

Every tool result carries human_review_required = True. These tools return
reference evidence only; a licensed clinician must review before any
patient-impacting decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow "import api.*" when launched by an MCP client from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from mcp.server.fastmcp import FastMCP

from api.config import ANTHROPIC_API_KEY, CONTACT, REQUEST_TIMEOUT
from api.crew import run_crew as _run_crew
from api.models import QueryRequest as _QueryRequest
from api.rag import get_index as _kb_index
from api.rag import reindex as _kb_reindex
from api.rag import search_knowledge_base as _kb_search
from api.retrieval import primary_term
from api.retrieval import normalize_drug as _rx_normalize
from api.retrieval import search_clinical_trials as _ctgov_search
from api.retrieval import medlineplus_connect_drug as _mp_connect
from api.retrieval import search_consumer_health as _medlineplus_search
from api.retrieval import search_drug_label as _fda_label
from api.retrieval import search_literature as _pubmed_search
from api.retrieval import search_myhealthfinder as _myhealthfinder
from api.retrieval import search_nhs_conditions as _nhs_conditions
from api.retrieval import surveillance_signal as _faers_trend

mcp = FastMCP("clinical-agent")

_HEADERS = {"User-Agent": f"clinical-agent-mcp/1.0 ({CONTACT})"}


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers=_HEADERS, follow_redirects=True)


def _shape(result: tuple[list[dict], str, str | None]) -> dict:
    """Turn a retrieval function's (raw_citations, source_api, note) into a plain dict."""
    raw_cites, source_api, note = result
    records = []
    for c in raw_cites:
        rec = {k: v for k, v in c.items() if not k.startswith("_")}
        if c.get("_detail"):
            rec["detail"] = c["_detail"]
        if c.get("_series"):
            rec["series"] = [{"period": p, "count": n} for p, n in c["_series"]]
        records.append(rec)
    return {
        "source_api": source_api,
        "note": note,
        "result_count": len(records),
        "results": records,
        "human_review_required": True,
    }


@mcp.tool()
async def search_local_guidelines(query: str, k: int = 4) -> dict:
    """Semantic search over the institution's local knowledge base (RAG corpus).

    Indexed documents under `knowledge/` — local protocols, formulary notes,
    guideline adaptations that are not on any public API. TF-IDF cosine retrieval,
    no embedding model or API key. Local policy overrides general sources when it
    applies, so call this first.

    Args:
        query: The clinical question or key terms.
        k: Passages to return, 1-8 (default 4).
    """
    return _shape(await _kb_search(query=query, k=int(k)))


@mcp.tool()
async def reindex_knowledge_base() -> dict:
    """Rebuild the local knowledge-base index from the files in `knowledge/`.

    Run after adding or editing documents. Returns document and chunk counts.
    """
    stats = _kb_reindex()
    return {**stats, "human_review_required": True}


@mcp.tool()
async def search_medical_literature(query: str, max_results: int = 6) -> dict:
    """Search PubMed for peer-reviewed medical literature.

    Use for treatment research, epidemiology / disease-surveillance context, and
    background for clinical narratives. Returns titles, authors, journal,
    publication date, PMID, and a link per article.

    Args:
        query: Search terms — a condition, drug, intervention, or clinical question.
        max_results: 1-10 articles to return (default 6).
    """
    n = max(1, min(int(max_results), 10))
    async with _new_client() as client:
        return _shape(await _pubmed_search(client, query=query, retmax=n))


@mcp.tool()
async def search_drug_label(drug_name: str) -> dict:
    """Look up FDA structured product labelling for a brand or generic drug name.

    Returns indications, dosage and administration, contraindications, boxed and
    other warnings, drug interactions, and use in specific populations, sourced
    live from openFDA. Prefer the generic name if a brand name returns nothing.
    """
    async with _new_client() as client:
        return _shape(await _fda_label(client, query=drug_name))


@mcp.tool()
async def normalize_drug_name(drug_name: str) -> dict:
    """Resolve a drug name to its standardized RxNorm concept.

    Returns the RxCUI, canonical name, term type, and synonyms. Use to
    disambiguate a drug name before other lookups or when reconciling
    medication lists.
    """
    async with _new_client() as client:
        return _shape(await _rx_normalize(client, drug_name=drug_name))


@mcp.tool()
async def surveillance_signal(term: str) -> dict:
    """Retrieve an FDA FAERS adverse-event report trend for a drug or reaction term.

    Returns yearly spontaneous-report volume as a time series — a
    pharmacovigilance / disease-surveillance signal. Spontaneous reports have no
    denominator and do not establish causation.

    Args:
        term: A drug generic name, or a MedDRA reaction term (e.g. "hepatotoxicity").
    """
    async with _new_client() as client:
        return _shape(await _faers_trend(client, term=term))


@mcp.tool()
async def search_clinical_trials(query: str, max_results: int = 6) -> dict:
    """Search ClinicalTrials.gov (API v2) for registered studies.

    Use for treatment research: what interventions are being or have been trialled
    for a condition or drug, plus phase, recruitment status, study type, sponsor,
    and enrollment. Returns the NCT id and a link per study.

    Args:
        query: Condition, drug, intervention, or free-text query.
        max_results: 1-20 studies to return (default 6), newest updates first.
    """
    async with _new_client() as client:
        return _shape(await _ctgov_search(client, query=query, page_size=int(max_results)))


@mcp.tool()
async def search_patient_education(term: str, max_results: int = 4) -> dict:
    """Look up MedlinePlus (NLM) plain-language health-topic pages.

    Consumer-facing information on conditions, symptoms, tests, and treatments —
    the authoritative open stand-in for sites like WebMD, which expose no public
    API. Written for patients and caregivers, not as clinical guidance.

    Args:
        term: A condition, symptom, test, or treatment.
        max_results: 1-10 topic pages to return (default 4).
    """
    async with _new_client() as client:
        return _shape(await _medlineplus_search(client, term=term, retmax=int(max_results)))


@mcp.tool()
async def search_prevention_guidance(term: str, max_results: int = 4) -> dict:
    """MyHealthfinder (U.S. health.gov) patient-facing prevention and screening
    guidance — what to do, when, and why. Free, no key.

    Args:
        term: A condition, screening topic, or life stage.
        max_results: 1-8 topics to return (default 4).
    """
    async with _new_client() as client:
        return _shape(await _myhealthfinder(client, term=term, limit=int(max_results)))


@mcp.tool()
async def search_nhs_conditions(term: str) -> dict:
    """NHS website (nhs.uk) UK conditions A-Z, patient-facing.

    Requires NHS_API_KEY in the server environment; returns a note (no error) when
    it is not configured.
    """
    async with _new_client() as client:
        return _shape(await _nhs_conditions(client, term=term))


@mcp.tool()
async def medlineplus_connect_drug(term: str, max_results: int = 4) -> dict:
    """MedlinePlus Connect patient-education entries for a drug, looked up by its
    RxNorm code (resolved from the drug name first).

    Args:
        term: A drug name.
        max_results: 1-6 entries to return (default 4).
    """
    async with _new_client() as client:
        return _shape(await _mp_connect(client, term=term, limit=int(max_results)))


@mcp.tool()
async def gather_clinical_evidence(query: str, mode: str = "decision_support") -> dict:
    """Run the workflow-appropriate retrieval plan for a clinical query in one call.

    Mirrors the Clinical Agent web app's evidence-only path: it picks which
    sources to query based on the workflow, runs them, and returns every source
    together with the search terms used (traceability).

    Args:
        query: The clinical question in free text.
        mode: One of "surveillance", "narratives", "treatment_research",
            "decision_support" (default).
    """
    mode = mode if mode in {"surveillance", "narratives", "treatment_research", "decision_support"} else "decision_support"
    term = primary_term(query)
    steps: list[dict] = []

    # Local knowledge base first, for every workflow.
    steps.append({"tool": "search_local_guidelines", "query": query,
                  **_shape(await _kb_search(query=query, k=4))})

    async with _new_client() as client:
        if mode == "surveillance":
            steps.append({"tool": "surveillance_signal", "term": term,
                          **_shape(await _faers_trend(client, term=term))})

        steps.append({"tool": "search_medical_literature", "query": query,
                      **_shape(await _pubmed_search(client, query=query, retmax=6))})

        if mode == "treatment_research":
            steps.append({"tool": "search_clinical_trials", "query": query,
                          **_shape(await _ctgov_search(client, query=query, page_size=6))})

        if mode in ("decision_support", "treatment_research"):
            steps.append({"tool": "search_drug_label", "drug_name": term,
                          **_shape(await _fda_label(client, query=term))})

        if mode == "decision_support":
            steps.append({"tool": "normalize_drug_name", "drug_name": term,
                          **_shape(await _rx_normalize(client, drug_name=term))})
            steps.append({"tool": "medlineplus_connect_drug", "term": term,
                          **_shape(await _mp_connect(client, term=term, limit=4))})

        if mode == "narratives":
            steps.append({"tool": "search_patient_education", "term": term,
                          **_shape(await _medlineplus_search(client, term=term, retmax=4))})
            steps.append({"tool": "search_prevention_guidance", "term": term,
                          **_shape(await _myhealthfinder(client, term=term, limit=4))})
            steps.append({"tool": "search_nhs_conditions", "term": term,
                          **_shape(await _nhs_conditions(client, term=term))})

    return {
        "mode": mode,
        "query": query,
        "primary_term_used": term,
        "steps": steps,
        "total_sources": sum(s["result_count"] for s in steps),
        "human_review_required": True,
        "disclaimer": "Reference evidence only. A licensed clinician must review "
                      "before any patient-impacting decision.",
    }


@mcp.tool()
async def run_clinical_crew(query: str, mode: str = "decision_support") -> dict:
    """Run the multi-agent thought generator (Planner → Researcher → Synthesist →
    Safety Reviewer) and return the full chain of thoughts with the cited answer.

    Requires ANTHROPIC_API_KEY in this server's environment (every crew agent is
    LLM-based). For retrieval only, use gather_clinical_evidence instead.

    Args:
        query: The clinical question.
        mode: "surveillance" | "narratives" | "treatment_research" | "decision_support".
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set in the MCP server environment; "
                         "the crew cannot run. Use gather_clinical_evidence for retrieval only."}
    mode = mode if mode in {"surveillance", "narratives", "treatment_research", "decision_support"} else "decision_support"
    result = await _run_crew(_QueryRequest(mode=mode, query=query))
    payload = result.model_dump()
    payload["human_review_required"] = True
    return payload


if __name__ == "__main__":
    mcp.run()
