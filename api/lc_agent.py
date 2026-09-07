"""LangChain / LangGraph evidence-collection engine.

An alternative to the hand-rolled loop in `api/agent.py`: the same retrieval +
RAG functions, wrapped as LangChain tools and driven by a LangGraph ReAct agent
(`create_react_agent`) over `ChatAnthropic`.

Optional. Needs `pip install -r requirements-langchain.txt` (pure Python, works on
3.14). `LANGCHAIN_AVAILABLE` is False and the endpoint returns 503 if it is not
installed.

Same response shape as `/api/query` (`QueryResponse`): the retrieval trace and
every citation still come from the shared `Retriever`, so provenance is identical
regardless of which engine ran.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from .agent import SYSTEM_PROMPT, Retriever, _dispatch
from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_WORKSPACE_ID,
    CLAUDE_MODEL,
    CONTACT,
    MAX_TOKENS,
    REQUEST_TIMEOUT,
)
from .models import LLMInfo, MODE_LABEL, QueryRequest, QueryResponse

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    LANGCHAIN_AVAILABLE = True
except ImportError:  # package set not installed
    LANGCHAIN_AVAILABLE = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _observation(out: dict) -> str:
    """Compact tool result for the model — keeps citation ids so it can cite [n]."""
    if "error" in out:
        return json.dumps(out)
    payload = {
        "note": out.get("note"),
        "results": [
            {"cite": r.get("cite"), "title": r.get("title"),
             "published": r.get("published"), "detail": r.get("detail")}
            for r in out.get("results", [])
        ],
    }
    return json.dumps(payload)[:8000]


def _build_tools(retr: Retriever):
    """LangChain tools that delegate to the shared retrieval dispatch and let the
    Retriever accumulate citations + trace."""

    @tool
    async def search_local_guidelines(query: str, k: int = 4) -> str:
        """Semantic search over the institution's local knowledge base (protocols,
        formulary notes, guideline excerpts not on any public API). Call this first."""
        return _observation(await _dispatch(retr, "search_local_guidelines", {"query": query, "k": k}))

    @tool
    async def search_medical_literature(query: str, max_results: int = 6) -> str:
        """Search PubMed for peer-reviewed literature. Use for treatment research,
        epidemiology, and background for clinical narratives."""
        return _observation(await _dispatch(retr, "search_medical_literature",
                                            {"query": query, "max_results": max_results}))

    @tool
    async def search_clinical_trials(query: str, max_results: int = 6) -> str:
        """Search ClinicalTrials.gov (API v2) for registered studies: phase,
        status, interventions, sponsor. Use for treatment research."""
        return _observation(await _dispatch(retr, "search_clinical_trials",
                                            {"query": query, "max_results": max_results}))

    @tool
    async def search_drug_label(drug_name: str) -> str:
        """FDA structured product labelling for a brand or generic drug name:
        indications, dosing, contraindications, warnings, interactions."""
        return _observation(await _dispatch(retr, "search_drug_label", {"drug_name": drug_name}))

    @tool
    async def normalize_drug_name(drug_name: str) -> str:
        """Resolve a drug name to its RxNorm concept (RxCUI, canonical name, synonyms)."""
        return _observation(await _dispatch(retr, "normalize_drug_name", {"drug_name": drug_name}))

    @tool
    async def surveillance_signal(term: str) -> str:
        """FDA FAERS adverse-event report volume over time for a drug name or a
        reaction term. Use for pharmacovigilance / surveillance signals."""
        return _observation(await _dispatch(retr, "surveillance_signal", {"term": term}))

    @tool
    async def search_patient_education(term: str) -> str:
        """MedlinePlus (NLM) plain-language health-topic pages on a condition,
        symptom, test, or treatment."""
        return _observation(await _dispatch(retr, "search_patient_education", {"term": term}))

    @tool
    async def search_prevention_guidance(term: str) -> str:
        """MyHealthfinder (U.S. health.gov) patient-facing prevention and screening
        guidance for a condition, screening topic, or life stage."""
        return _observation(await _dispatch(retr, "search_prevention_guidance", {"term": term}))

    @tool
    async def search_nhs_conditions(term: str) -> str:
        """NHS website (nhs.uk) UK conditions A-Z, patient-facing. Skipped with a
        note if the server has no NHS API key."""
        return _observation(await _dispatch(retr, "search_nhs_conditions", {"term": term}))

    @tool
    async def medlineplus_connect_drug(term: str) -> str:
        """MedlinePlus Connect patient-education entries for a drug, looked up by
        its RxNorm code."""
        return _observation(await _dispatch(retr, "medlineplus_connect_drug", {"term": term}))

    return [
        search_local_guidelines, search_medical_literature, search_clinical_trials,
        search_drug_label, normalize_drug_name, surveillance_signal, search_patient_education,
        search_prevention_guidance, search_nhs_conditions, medlineplus_connect_drug,
    ]


def _make_model(max_tokens: int | None = None) -> "ChatAnthropic":
    kwargs: dict = dict(
        model=CLAUDE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=max_tokens or MAX_TOKENS,
        timeout=60,
        max_retries=1,
    )
    if ANTHROPIC_WORKSPACE_ID:
        kwargs["default_headers"] = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID}
    return ChatAnthropic(**kwargs)


def _final_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # list of content blocks
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content).strip()


async def run_langchain_query(req: QueryRequest) -> QueryResponse:
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain engine not installed. pip install -r requirements-langchain.txt")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("LangChain engine requires ANTHROPIC_API_KEY.")

    from .agent import attach_fhir_context
    await attach_fhir_context(req)

    prompt = f"Workflow: {MODE_LABEL[req.mode]}\n\nClinician query:\n{req.query}"
    if req.case_context:
        prompt += f"\n\nDe-identified case context:\n{req.case_context}"
    prompt += ("\n\nRetrieve current evidence with your tools, then answer. "
               "Cite every clinical statement with [n] using the 'cite' ids from tool results.")

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": f"clinical-agent-langchain/1.0 ({CONTACT})"},
        follow_redirects=True,
    ) as client:
        retr = Retriever(client)
        agent = create_react_agent(_make_model(), _build_tools(retr), prompt=SYSTEM_PROMPT)
        state = await agent.ainvoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": 18},
        )
        messages = state["messages"]
        rounds = sum(1 for m in messages if getattr(m, "tool_calls", None))
        answer = _final_text(messages[-1]) or "_The agent returned no text._"

        return QueryResponse(
            mode=req.mode,
            query=req.query,
            answer_markdown=answer,
            citations=retr.citations,
            retrieval_trace=retr.trace,
            llm=LLMInfo(used=True, model=f"{CLAUDE_MODEL} (langchain/langgraph)", tool_rounds=rounds),
            human_review_required=True,
            generated_at=_now_iso(),
        )
