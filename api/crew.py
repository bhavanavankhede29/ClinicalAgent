"""Thought generator — a small multi-agent reasoning layer.

Self-contained, no agent framework. Primitives:

    Agent   — a persona with role / goal / backstory, backed by one Claude call
    Task    — a unit of work assigned to an agent, fed the prior tasks' output
    Crew    — runs the tasks in sequence
    step callback — every agent step is emitted as a "thought" for the trace

The clinical crew:

    1. Planner        decides which evidence tools to run and with what terms
    2. Researcher     executes the plan over the real retrieval + RAG tools
    3. Synthesist     writes the [n]-cited answer from what was retrieved
    4. Safety Reviewer checks it against the safety rules; one revision loop
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import httpx

from .agent import TOOLS, Retriever, _call_claude, _dispatch
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CONTACT, REQUEST_TIMEOUT
from .models import Citation, LLMInfo, MODE_LABEL, QueryRequest, ThinkResponse, ThoughtItem
from .retrieval import primary_term

# Which free-text argument each tool expects, so a plan step that omits it can be
# repaired rather than firing a lookup with an empty string.
_TOOL_ARG = {
    "search_local_guidelines": "query",
    "search_medical_literature": "query",
    "search_clinical_trials": "query",
    "search_drug_label": "drug_name",
    "normalize_drug_name": "drug_name",
    "surveillance_signal": "term",
    "search_patient_education": "term",
}

MAX_REVISIONS = 1

_SAFETY_CLAUSE = (
    "This is decision support for a licensed clinician, never a patient-facing "
    "directive. Do not give a definitive diagnosis or a patient-specific dose. "
    "Ground claims in retrieved sources; flag anything ungrounded. Human review "
    "is required before any patient-impacting action."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_block(text: str) -> dict:
    """Best-effort extraction of the first JSON object in a model reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _text_of(response: dict) -> str:
    return "\n".join(
        b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"
    ).strip()


# --- multi-agent primitives ---------------------------------------------------

@dataclass
class Agent:
    role: str
    goal: str
    backstory: str

    def system(self) -> str:
        return (f"You are the {self.role} on a clinical evidence crew.\n"
                f"Goal: {self.goal}\n"
                f"Background: {self.backstory}\n\n{_SAFETY_CLAUSE}")


@dataclass
class Thought:
    step: int
    agent: str
    kind: str
    content: str = ""
    data: dict | None = None


@dataclass
class ThoughtLog:
    items: list[Thought] = field(default_factory=list)

    def emit(self, agent: str, kind: str, content: str = "", data: dict | None = None) -> Thought:
        t = Thought(len(self.items) + 1, agent, kind, content.strip(), data)
        self.items.append(t)
        return t

    def as_models(self) -> list[ThoughtItem]:
        return [ThoughtItem(**asdict(t)) for t in self.items]


# --- the clinical crew --------------------------------------------------------

PLANNER = Agent(
    role="Retrieval Planner",
    goal=("Turn the clinician's question into a short ordered list of evidence "
          "lookups, choosing from the available tools and writing precise search terms."),
    backstory=("You know each source's strengths: the local knowledge base holds "
               "institutional policy and is checked first; PubMed for evidence; "
               "ClinicalTrials.gov for the trial landscape; openFDA label for dosing "
               "and warnings; RxNorm to disambiguate a drug; FAERS for safety signals; "
               "MedlinePlus for patient-facing wording."),
)

RESEARCHER = Agent(
    role="Evidence Researcher",
    goal="Run the plan exactly, and note what each source did and did not return.",
    backstory="You never invent findings. A tool that returns nothing is itself a result.",
)

SYNTHESIST = Agent(
    role="Clinical Synthesist",
    goal=("Write a concise, scannable answer in Markdown, citing every clinical "
          "statement with [n] against the numbered sources."),
    backstory=("You present options and tradeoffs from the evidence, surface "
               "contraindications and boxed warnings first, and state plainly when "
               "the evidence is thin."),
)

REVIEWER = Agent(
    role="Safety Reviewer",
    goal=("Audit the draft: every clinical claim carries a [n]; no patient-specific "
          "dose or definitive diagnosis is asserted as a directive; contraindications "
          "and uncertainty are visible; the closing review line is present."),
    backstory="You return strict JSON with a verdict and specific, actionable issues.",
)


async def _run_agent(client, agent: Agent, user_prompt: str, *, max_tokens: int = 1400) -> str:
    resp = await _call_claude(
        client, [{"role": "user", "content": user_prompt}],
        use_tools=False, system=agent.system(), max_tokens=max_tokens,
    )
    return _text_of(resp)


def _tool_catalogue() -> str:
    return "\n".join(f"- {t['name']}: {t['description']}" for t in TOOLS)


def _render_citations(cites: list[Citation]) -> str:
    if not cites:
        return "(no sources retrieved)"
    return "\n".join(
        f"[{c.id}] ({c.source_kind}) {c.title} — {c.source}\n    {c.snippet}"
        for c in cites
    )


async def run_crew(req: QueryRequest) -> ThinkResponse:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Thought generator requires ANTHROPIC_API_KEY — every crew agent is LLM-based.")

    from .agent import attach_fhir_context
    await attach_fhir_context(req)

    log = ThoughtLog()
    llm_calls = 0
    revisions = 0

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": f"clinical-agent-crew/1.0 ({CONTACT})"},
        follow_redirects=True,
    ) as client:
        retr = Retriever(client)
        context = f"Workflow: {MODE_LABEL[req.mode]}\nClinician question: {req.query}"
        if req.case_context:
            context += f"\nDe-identified case context: {req.case_context}"

        # 1 — Planner ------------------------------------------------------
        planner_reply = await _run_agent(client, PLANNER, (
            f"{context}\n\nAvailable tools:\n{_tool_catalogue()}\n\n"
            "Respond with JSON only:\n"
            '{"rationale": "<2-3 sentences>", "steps": [{"tool": "<name>", '
            '"args": {<tool args>}, "why": "<short>"}]}\n'
            "Always include search_local_guidelines first. 2-5 steps total."
        ))
        llm_calls += 1
        plan = _json_block(planner_reply)
        steps = plan.get("steps") or [{"tool": "search_local_guidelines", "args": {"query": req.query}}]
        log.emit(PLANNER.role, "plan", plan.get("rationale", planner_reply),
                 data={"steps": steps})
        for s in steps:
            log.emit(PLANNER.role, "thought",
                     f"{s.get('tool')} — {s.get('why', '')}".strip(" —"), data=s)

        # 2 — Researcher -------------------------------------------------
        if not any(s.get("tool") == "search_local_guidelines" for s in steps):
            steps.insert(0, {"tool": "search_local_guidelines", "args": {"query": req.query}})
        for s in steps:
            tool = s.get("tool", "")
            args = dict(s.get("args") or {})
            arg_key = _TOOL_ARG.get(tool)
            if arg_key and not str(args.get(arg_key, "")).strip():
                args[arg_key] = req.query if arg_key == "query" else primary_term(req.query)
            log.emit(RESEARCHER.role, "action", f"call {tool}", data={"tool": tool, "args": args})
            out = await _dispatch(retr, tool, args)
            if "error" in out:
                log.emit(RESEARCHER.role, "observation", f"{tool}: {out['error']}")
                continue
            found = out.get("results", [])
            titles = "; ".join(r.get("title", "?") for r in found[:4]) or "no results"
            note = f" ({out['note']})" if out.get("note") else ""
            log.emit(RESEARCHER.role, "observation",
                     f"{tool}: {len(found)} result(s){note} — {titles}",
                     data={"citation_ids": out.get("citation_ids", [])})

        citations = retr.citations
        log.emit(RESEARCHER.role, "decision",
                 f"{len(citations)} source(s) gathered across {len(retr.trace)} lookup(s).")

        # 3 — Synthesist ----------------------------------------------
        sources_text = _render_citations(citations)
        draft = await _run_agent(client, SYNTHESIST, (
            f"{context}\n\nNumbered sources:\n{sources_text}\n\n"
            "Write the answer now. Markdown only. Cite every clinical statement with [n]. "
            'End with exactly: "Clinician review required before any patient-impacting action."'
        ))
        llm_calls += 1
        log.emit(SYNTHESIST.role, "draft", draft)

        # 4 — Safety Reviewer + one revision loop --------------------
        for _ in range(MAX_REVISIONS + 1):
            review_reply = await _run_agent(client, REVIEWER, (
                f"Clinician question: {req.query}\n\nNumbered sources:\n{sources_text}\n\n"
                f"Draft to audit:\n---\n{draft}\n---\n\n"
                'Respond with JSON only: {"verdict": "approved" | "revise", '
                '"issues": ["<specific fix>", ...], "note": "<one line>"}'
            ), max_tokens=600)
            llm_calls += 1
            review = _json_block(review_reply)
            verdict = (review.get("verdict") or "approved").lower()
            issues = review.get("issues") or []
            critique = review.get("note") or review_reply.strip()
            if issues:
                critique = (critique + "\n" if critique else "") + "\n".join(f"• {i}" for i in issues)
            critique = critique or f"verdict: {verdict}"
            log.emit(REVIEWER.role, "critique", critique,
                     data={"verdict": verdict, "issues": issues})
            if verdict != "revise" or not issues:
                break
            revisions += 1
            draft = await _run_agent(client, SYNTHESIST, (
                f"{context}\n\nNumbered sources:\n{sources_text}\n\n"
                f"Your previous draft:\n---\n{draft}\n---\n\n"
                f"The reviewer requires these fixes:\n- " + "\n- ".join(issues) + "\n\n"
                "Return the corrected answer, same rules (Markdown, [n] citations, closing line)."
            ))
            llm_calls += 1
            log.emit(SYNTHESIST.role, "draft", draft)

        closing = "Clinician review required before any patient-impacting action."
        if closing.lower() not in draft.lower():
            draft = draft.rstrip() + f"\n\n{closing}"

        return ThinkResponse(
            mode=req.mode,
            query=req.query,
            thoughts=log.as_models(),
            answer_markdown=draft,
            citations=citations,
            retrieval_trace=retr.trace,
            llm=LLMInfo(used=True, model=CLAUDE_MODEL, tool_rounds=llm_calls),
            revisions=revisions,
            human_review_required=True,
            generated_at=_now_iso(),
        )


# --- CLI --------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(description="Run the clinical thought-generator crew.")
    ap.add_argument("query")
    ap.add_argument("--mode", default="decision_support",
                    choices=list(MODE_LABEL.keys()))
    ap.add_argument("--context", default=None)
    args = ap.parse_args()

    result = asyncio.run(run_crew(QueryRequest(
        mode=args.mode, query=args.query, case_context=args.context,
    )))
    for t in result.thoughts:
        head = f"[{t.step}] {t.agent} · {t.kind}"
        print(f"\n{head}\n{'-' * len(head)}\n{t.content}")
    print("\n\n=== ANSWER ===\n")
    print(result.answer_markdown)
    print(f"\n({len(result.citations)} citations · {result.revisions} revision(s) · "
          f"{result.llm.tool_rounds} LLM calls)")


if __name__ == "__main__":
    _main()
