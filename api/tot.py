"""Tree-of-Thoughts reasoning engine.

Where ``/api/query`` runs a single tool-use chain and ``/api/query/rag-graph``
runs a fixed corrective-RAG graph, this explores several candidate *approaches*
to the question in parallel, scores them against retrieved evidence, keeps the
strongest, deepens those, and synthesises one cited answer from the winning
path.

    propose 3 approaches
        └─ for each: targeted retrieval → draft partial answer
             └─ evaluate all drafts (0-10, grounded in evidence?)
                  └─ keep top 1-2 → deepen each
                       └─ synthesise final answer from the survivors

Breadth 3, depth 2, keep top 2 (top 1 if the runner-up trails badly). ~7-9
LLM calls. Response shape is the shared ``QueryResponse``; the tree summary is
in ``llm.reason``. Needs ``ANTHROPIC_API_KEY``; no extra packages.
"""
from __future__ import annotations

import asyncio
import json
import re

from .agent import (
    SYSTEM_PROMPT,
    Retriever,
    _call_claude,
    _COMPREHENSIVE_INSTRUCTION,
    _deterministic_plan,
    _local_last,
    _now_iso,
    _TREATMENT_PLAN_INSTRUCTION,
    attach_fhir_context,
    looks_like_bare_toolcall,
    missing_plan_context,
    strip_preamble,
)
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS
from .models import LLMInfo, MODE_LABEL, QueryRequest, QueryResponse
from .net import get_client

BRANCHES = 3          # approaches proposed at ply 0
KEEP = 2              # survivors carried into the deepen ply
KEEP_GAP = 3          # drop the runner-up if it trails the leader by more than this


def _text_of(resp: dict) -> str:
    return "\n".join(
        b["text"] for b in resp.get("content", []) if b.get("type") == "text"
    ).strip()


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
        return "(no sources retrieved yet)"
    return "\n".join(
        f"[{c.id}] ({c.source_kind}) {c.title} — {c.source}\n    {c.snippet}"
        for c in citations
    )


def _final_instruction(mode: str) -> str:
    if mode == "comprehensive":
        return _COMPREHENSIVE_INSTRUCTION
    if mode == "treatment_plan":
        return _TREATMENT_PLAN_INSTRUCTION
    return ("\n\nAnswer from the numbered sources only. Cite every clinical statement "
            "with [n]. If the evidence is thin, say so in one clause rather than filling "
            "the gap. Do not name tools or narrate which searches returned nothing; no preamble.")


async def run_tot(req: QueryRequest) -> QueryResponse:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Tree-of-Thoughts engine requires ANTHROPIC_API_KEY.")

    await attach_fhir_context(req)
    guard = missing_plan_context(req)
    if guard is not None:
        return guard

    client = get_client()
    retr = Retriever(client)
    ctx = f"\n\nDe-identified case context:\n{req.case_context}" if req.case_context else ""
    question = f"Workflow: {MODE_LABEL[req.mode]}\n\nClinician question:\n{req.query}{ctx}"
    calls = 0

    # ---- ply 0: propose distinct approaches -------------------------------------
    calls += 1
    prop = await _call_claude(
        client,
        [{"role": "user", "content":
            f"{question}\n\nPropose exactly {BRANCHES} genuinely different approaches to "
            "answering this — different framings, evidence angles, or decision lenses, not "
            "restatements. For each give a one-sentence framing and a focused search query "
            "built from the CLINICAL content of the case (conditions, drugs, labs) — never "
            "the workflow name or a phrase like 'treatment plan'. "
            'Reply JSON only: {"approaches":[{"framing":"...","search":"..."}]}'}],
        use_tools=False,
        system="You plan distinct lines of clinical reasoning. Reply with JSON only.",
        max_tokens=600,
    )
    approaches = _json_block(_text_of(prop)).get("approaches") or []
    approaches = [a for a in approaches if isinstance(a, dict) and a.get("search")][:BRANCHES]

    # A search term to fall back on. For treatment_plan mode req.query is a
    # placeholder ("Personalized treatment plan…"), so the real clinical content
    # is in case_context — seed retrieval from that, not the placeholder.
    seed = (req.case_context or req.query).strip()[:280] or req.query
    if not approaches:
        approaches = [{"framing": "Direct evidence review", "search": seed}]

    # ---- ply 1: retrieve. Always do one baseline pass on the case itself so the
    #      shared Retriever has the core evidence even if an approach's search
    #      string is weak, then one pass per approach. (Sequential — it mutates
    #      the shared Retriever; drafting below runs concurrently.) -------------
    await _deterministic_plan(retr, QueryRequest(mode=req.mode, query=seed))
    for a in approaches:
        q = str(a.get("search") or "").strip()[:300] or seed
        await _deterministic_plan(retr, QueryRequest(mode=req.mode, query=q))

    async def _draft(a: dict) -> str:
        nonlocal calls
        calls += 1
        r = await _call_claude(
            client,
            [{"role": "user", "content":
                f"{question}\n\nApproach: {a.get('framing', '')}\n\n"
                f"Numbered sources:\n{_render_context(retr.citations)}\n\n"
                "Draft concise notes answering the question strictly from this approach and "
                "these sources. Cite [n]. If this approach's evidence is thin, note it in a "
                "clause — do not name tools or describe your searches."}],
            use_tools=False,
            system=SYSTEM_PROMPT,
            max_tokens=900,
        )
        return _text_of(r) or "(no draft)"

    drafts = await asyncio.gather(*(_draft(a) for a in approaches))

    # ---- ply 1.5: evaluate -----------------------------------------------------
    calls += 1
    blocks = "\n\n".join(
        f"[{i}] Framing: {approaches[i].get('framing', '')}\n{drafts[i]}" for i in range(len(drafts))
    )
    ev = await _call_claude(
        client,
        [{"role": "user", "content":
            f"Question:\n{req.query}\n\nCandidate drafts:\n{blocks}\n\n"
            "Score each 0-10 on: grounded in the cited evidence, answers the question, "
            "clinically sound, surfaces contraindications/uncertainty. "
            'Reply JSON only: {"scores":[{"i":0,"score":0,"why":"one line"}]}'}],
        use_tools=False,
        system="You are a careful clinical reviewer. Reply with JSON only.",
        max_tokens=500,
    )
    scored = _json_block(_text_of(ev)).get("scores") or []
    ranked = sorted(
        ({"i": int(s["i"]), "score": float(s.get("score", 0)), "why": str(s.get("why", ""))[:160]}
         for s in scored if isinstance(s, dict) and "i" in s and 0 <= int(s["i"]) < len(drafts)),
        key=lambda s: s["score"], reverse=True,
    ) or [{"i": i, "score": 0.0, "why": ""} for i in range(len(drafts))]

    survivors = ranked[:KEEP]
    if len(survivors) == 2 and survivors[0]["score"] - survivors[1]["score"] > KEEP_GAP:
        survivors = survivors[:1]

    # ---- ply 2: deepen each survivor ----------------------------------------------
    async def _deepen(s: dict) -> str:
        nonlocal calls
        calls += 1
        i = s["i"]
        r = await _call_claude(
            client,
            [{"role": "user", "content":
                f"{question}\n\nYour earlier draft:\n{drafts[i]}\n\n"
                f"Reviewer note: {s['why']}\n\n"
                f"Numbered sources:\n{_render_context(retr.citations)}\n\n"
                "Produce a stronger version: close the single biggest gap the reviewer "
                "flagged, keep every claim cited [n], and stay within this evidence."}],
            use_tools=False,
            system=SYSTEM_PROMPT,
            max_tokens=1100,
        )
        return _text_of(r) or drafts[i]

    deepened = await asyncio.gather(*(_deepen(s) for s in survivors))

    # ---- synthesise ------------------------------------------------------------
    calls += 1
    dev = "\n\n---\n\n".join(deepened)
    answer_tokens = 3600 if req.mode in ("comprehensive", "treatment_plan") else MAX_TOKENS
    syn = await _call_claude(
        client,
        [{"role": "user", "content":
            f"{question}\n\nYou explored {len(approaches)} approaches; the strongest, "
            f"developed:\n\n{dev}\n\n"
            f"Numbered sources:\n{_render_context(retr.citations)}"
            f"{_final_instruction(req.mode)}"}],
        use_tools=False,
        system=SYSTEM_PROMPT,
        max_tokens=answer_tokens,
    )
    answer = _text_of(syn) or "_The model returned no text._"
    if looks_like_bare_toolcall(answer):
        calls += 1
        syn = await _call_claude(
            client,
            [{"role": "user", "content":
                f"{question}\n\nNumbered sources:\n{_render_context(retr.citations)}\n\n"
                "Write the answer in prose now, strictly from these sources, citing [n]. "
                "Do not output a tool call or function-call syntax."
                f"{_final_instruction(req.mode)}"}],
            use_tools=False,
            system=SYSTEM_PROMPT,
            max_tokens=answer_tokens,
        )
        answer = _text_of(syn) or answer

    answer = strip_preamble(answer)
    if not answer or looks_like_bare_toolcall(answer):
        # The model would not produce prose. Fall back to an organised list of
        # everything the tree retrieved rather than showing a tool call.
        from .agent import _evidence_only_answer
        answer = _evidence_only_answer(req, retr, intro=(
            "**Automated synthesis was unavailable for this request** — the reasoning "
            "engine did not return a written answer. The sources gathered across every "
            "branch are listed below for clinician synthesis."
        ))

    citations, answer = _local_last(retr.citations, answer)
    kept = ", ".join(f"#{s['i']}({s['score']:.0f})" for s in survivors)
    all_scores = ", ".join(f"#{s['i']}:{s['score']:.0f}" for s in ranked)
    return QueryResponse(
        mode=req.mode,
        query=req.query,
        answer_markdown=answer,
        citations=citations,
        retrieval_trace=retr.trace,
        llm=LLMInfo(
            used=True,
            model=f"{CLAUDE_MODEL} (tree-of-thoughts)",
            tool_rounds=calls,
            reason=f"{len(approaches)} approaches [{all_scores}] → kept {kept} → synthesised "
                   f"from {len(citations)} sources",
        ),
        human_review_required=True,
        generated_at=_now_iso(),
    )
