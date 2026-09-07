"""Corrective-RAG pipeline built as an explicit LangGraph StateGraph.

Unlike the ReAct agent in ``lc_agent.py`` (which exposes retrieval as tools and
lets the model decide), this is a fixed graph:

    START → retrieve → grade ──sufficient──▶ generate → END
                         │
                     insufficient
                         ▼
                      rewrite ──▶ (back to retrieve, once)

* retrieve  — runs the workflow's deterministic source plan (live APIs + the
  local knowledge base / Voyage vector search) into one shared Retriever.
* grade     — an LLM check on whether the retrieved context can answer the
  question.
* rewrite   — if not, an LLM rewrites the search query and we retrieve once more.
* generate  — the LLM writes the cited answer from the numbered context.

Response shape is the shared ``QueryResponse``; provenance (citations, retrieval
trace) is identical to ``/api/query``. Optional — needs the langchain extras.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from .agent import _COMPREHENSIVE_INSTRUCTION, SYSTEM_PROMPT, Retriever, _deterministic_plan, _local_last
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .models import LLMInfo, MODE_LABEL, QueryRequest, QueryResponse
from .net import get_client

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph

    from .lc_agent import LANGCHAIN_AVAILABLE, _make_model

    RAG_GRAPH_AVAILABLE = LANGCHAIN_AVAILABLE
except ImportError:  # langchain extras not installed
    RAG_GRAPH_AVAILABLE = False

MAX_REWRITES = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _msg_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text").strip()
    return str(content).strip()


def _json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _render_context(citations) -> str:
    if not citations:
        return "(no sources retrieved)"
    return "\n".join(
        f"[{c.id}] ({c.source_kind}) {c.title} — {c.source}\n    {c.snippet}"
        for c in citations
    )


class RagState(TypedDict, total=False):
    search_query: str
    rewrites: int
    context: str
    grade: str
    missing: str
    answer: str


async def run_rag_graph(req: QueryRequest) -> QueryResponse:
    if not RAG_GRAPH_AVAILABLE:
        raise RuntimeError("LangGraph RAG engine not installed. pip install -r requirements-langchain.txt")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("LangGraph RAG engine requires ANTHROPIC_API_KEY.")

    from .agent import attach_fhir_context
    await attach_fhir_context(req)
    _ctx = f"\n\nDe-identified case context:\n{req.case_context}" if req.case_context else ""

    retr = Retriever(get_client())          # one shared Retriever → citations accumulate across retries
    grader = _make_model(max_tokens=400)
    writer = _make_model(max_tokens=3600 if req.mode == "comprehensive" else 1600)
    llm_calls = 0

    async def retrieve(state: RagState) -> RagState:
        sq = state.get("search_query") or req.query
        await _deterministic_plan(retr, QueryRequest(mode=req.mode, query=sq))
        return {"context": _render_context(retr.citations)}

    async def grade(state: RagState) -> RagState:
        nonlocal llm_calls
        llm_calls += 1
        reply = await grader.ainvoke([
            SystemMessage(content="You judge whether retrieved sources are enough to answer a "
                                  "clinician's question. Reply with JSON only."),
            HumanMessage(content=f"Question: {req.query}\n\nRetrieved sources:\n{state['context']}\n\n"
                                 'Reply: {"sufficient": true|false, "missing": "<one line: what is missing>"}'),
        ])
        data = _json_block(_msg_text(reply))
        return {"grade": "sufficient" if data.get("sufficient", True) else "insufficient",
                "missing": str(data.get("missing", ""))[:200]}

    async def rewrite(state: RagState) -> RagState:
        nonlocal llm_calls
        llm_calls += 1
        reply = await grader.ainvoke([
            SystemMessage(content="Rewrite a clinical search query to retrieve the missing evidence. "
                                  "Reply with only the new query text."),
            HumanMessage(content=f"Question: {req.query}\nMissing: {state.get('missing', '')}\n"
                                 f"Previous query: {state.get('search_query') or req.query}\nNew query:"),
        ])
        new_q = _msg_text(reply).strip().strip('"')[:300] or req.query
        return {"search_query": new_q, "rewrites": state.get("rewrites", 0) + 1}

    async def generate(state: RagState) -> RagState:
        nonlocal llm_calls
        llm_calls += 1
        instruction = (_COMPREHENSIVE_INSTRUCTION if req.mode == "comprehensive"
                       else "\n\nAnswer from the numbered sources. Cite every clinical statement with [n].")
        reply = await writer.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Workflow: {MODE_LABEL[req.mode]}\n\nClinician question:\n{req.query}{_ctx}\n\n"
                                 f"Numbered sources:\n{state['context']}{instruction}"),
        ])
        return {"answer": _msg_text(reply) or "_The model returned no text._"}

    def route(state: RagState) -> str:
        if state.get("grade") == "sufficient" or state.get("rewrites", 0) >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route, {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    compiled = graph.compile()

    final = await compiled.ainvoke(
        {"search_query": req.query, "rewrites": 0},
        config={"recursion_limit": 12},
    )

    citations, answer = _local_last(retr.citations, final.get("answer", ""))
    return QueryResponse(
        mode=req.mode,
        query=req.query,
        answer_markdown=answer,
        citations=citations,
        retrieval_trace=retr.trace,
        llm=LLMInfo(
            used=True,
            model=f"{CLAUDE_MODEL} (langgraph corrective-RAG)",
            tool_rounds=llm_calls,
            reason=f"grade={final.get('grade', 'n/a')}, rewrites={final.get('rewrites', 0)}",
        ),
        human_review_required=True,
        generated_at=_now_iso(),
    )
