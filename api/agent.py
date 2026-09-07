"""Orchestration.

Two paths, one response shape:

* LLM path (ANTHROPIC_API_KEY set) — Claude runs a tool-use loop over the
  retrieval functions and writes a cited synthesis.
* Evidence-only path — a deterministic retrieval plan runs and the sources are
  organised without interpretation.

Either way the caller gets the full retrieval trace, every citation with its
origin API and retrieval time, and human_review_required = True.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone

import httpx

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_WORKSPACE_ID,
    CLAUDE_MODEL,
    CONTACT,
    MAX_TOKENS,
    MAX_TOOL_ROUNDS,
    REQUEST_TIMEOUT,
)
from .models import Citation, LLMInfo, MODE_LABEL, QueryRequest, QueryResponse, TraceStep
from .net import cached_call, get_client, with_retry
from .rag import search_knowledge_base
from .retrieval import (
    medlineplus_connect_drug,
    normalize_drug,
    primary_term,
    search_clinical_trials,
    search_consumer_health,
    search_drug_label,
    search_literature,
    search_myhealthfinder,
    search_nhs_conditions,
    surveillance_signal,
    surveillance_top_reported,
)

SYSTEM_PROMPT = """You are Clinical Agent, an evidence-grounded decision-support assistant for licensed healthcare professionals.

Operating rules:
- Ground every clinical statement in a source you retrieved this turn and attach a [n] citation using the citation ids from tool results. If you cannot ground a statement, do not make it.
- Consult the local knowledge base with search_local_guidelines. When an indexed institutional protocol addresses the question, it takes precedence over general sources; cite it and note where it diverges from them. List local-knowledge citations after the external sources.
- Prefer a tool call over recall for anything about drugs, dosing, interactions, epidemiology, or current practice.
- If the tools return nothing useful, say so plainly and stop. Do not fill the gap with unverified knowledge.
- Be economical: 2-4 tool calls is normally enough. Once you have relevant sources, stop retrieving and write the answer.
- Do not issue a definitive diagnosis or a patient-specific dose or therapy directive. Present evidence-based options and the tradeoffs each source describes.
- Surface contraindications, boxed warnings, and uncertainty prominently. Start any genuinely critical safety item — a contraindication, a boxed / black-box warning, a life-threatening interaction, or a red-flag finding that needs urgent action — with a leading "⚠ " so it can be highlighted. Use it sparingly, only for true safety-critical points.
- Be concise and scannable: short sections, bullets, Markdown only.
- Do NOT name your tools, describe your retrieval process, or add a section about which searches returned nothing (e.g. never write "a search of the local knowledge base returned…", "search_local_guidelines found no match", or "the retrieved sources do not contain…"). Just write the grounded answer. If nothing relevant was found for the whole question, say that in one sentence and stop.
- Never emit function-call syntax (e.g. `search_local_guidelines(query="…")`) as your reply. Call tools only through the tool interface; your text output is always prose for the clinician.
- Do not open with a "note on data completeness" preamble; if context is thin, work it into the relevant point in one short clause.
- Close with the single line: "Clinician review required before any patient-impacting action."
"""

TOOLS = [
    {
        "name": "search_local_guidelines",
        "description": "Semantic search over the institution's own indexed documents (local knowledge base): protocols, formulary notes, order-set rationale, guideline excerpts not on any public API. Local policy overrides general sources when it applies; cite these after the external sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The clinical question or key terms."},
                "k": {"type": "integer", "description": "Passages to return, 1-8, default 4."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_medical_literature",
        "description": "Search PubMed for peer-reviewed literature. Use for treatment research, epidemiology/surveillance context, and background for clinical narratives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms: condition, drug, or clinical question."},
                "max_results": {"type": "integer", "description": "1-10, default 6."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_clinical_trials",
        "description": "Search ClinicalTrials.gov (API v2) for registered interventional/observational studies: phase, recruitment status, interventions, sponsor. Use for treatment research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-20, default 6."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_drug_label",
        "description": "FDA structured product labelling for a brand or generic drug name: indications, dosing, contraindications, warnings, interactions, use in specific populations.",
        "input_schema": {
            "type": "object",
            "properties": {"drug_name": {"type": "string"}},
            "required": ["drug_name"],
        },
    },
    {
        "name": "normalize_drug_name",
        "description": "Resolve a drug name to its RxNorm concept (RxCUI, canonical name, synonyms). Use to disambiguate before other lookups.",
        "input_schema": {
            "type": "object",
            "properties": {"drug_name": {"type": "string"}},
            "required": ["drug_name"],
        },
    },
    {
        "name": "surveillance_signal",
        "description": "FDA FAERS adverse-event report volume over time for a drug name or a reaction term. Use for pharmacovigilance and disease-surveillance signals.",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    },
    {
        "name": "surveillance_top_reported",
        "description": "FDA FAERS: the most frequently reported adverse-event/reaction terms across ALL reports, no drug or condition named. Use this — not surveillance_signal — when the clinician asks a general question like 'list of diseases/reactions reported' with no specific drug, condition, or reaction to anchor on. These are reaction terms tied to drug-exposure reports, not a population disease registry.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "How many top terms to return (default 15, max 30)."}},
            "required": [],
        },
    },
    {
        "name": "search_patient_education",
        "description": "MedlinePlus (NLM) plain-language health-topic pages on a condition, symptom, test, or treatment. Use for patient-facing wording in a clinical narrative or handout.",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    },
    {
        "name": "search_prevention_guidance",
        "description": "MyHealthfinder (U.S. health.gov) patient-facing prevention and screening guidance — what to do, when, and why.",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    },
    {
        "name": "search_nhs_conditions",
        "description": "NHS website (nhs.uk) UK conditions A-Z, patient-facing. Requires an NHS API key on the server; returns a note if unavailable.",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    },
    {
        "name": "medlineplus_connect_drug",
        "description": "MedlinePlus Connect patient-education entries for a drug, looked up by its RxNorm code. Use for drug-specific patient information.",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string", "description": "Drug name."}},
            "required": ["term"],
        },
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TOOL_NAMES = {t["name"] for t in TOOLS}
_BARE_CALL_RE = re.compile(r"^([a-z_]+)\s*\(.*\)\s*$", re.IGNORECASE | re.DOTALL)


def looks_like_bare_toolcall(text: str) -> bool:
    """True when the model wrote a tool call as prose (e.g.
    `search_local_guidelines(query="…")`) instead of invoking it — that stray
    line then comes back as the whole answer. Callers recover by re-prompting."""
    if not text:
        return False
    s = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    s = s.strip("`").strip()
    if len(s) > 240:
        return False
    m = _BARE_CALL_RE.match(s)
    return bool(m and m.group(1) in _TOOL_NAMES)


# Which argument name each tool's primary string maps to, for salvaging a
# tool call the model wrote as prose instead of invoking.
_PRIMARY_ARG = {
    "search_local_guidelines": "query",
    "search_medical_literature": "query",
    "search_clinical_trials": "query",
    "search_drug_label": "drug_name",
    "normalize_drug_name": "drug_name",
    "surveillance_signal": "term",
    "search_patient_education": "term",
    "search_prevention_guidance": "term",
    "search_nhs_conditions": "term",
    "medlineplus_connect_drug": "term",
}


def parse_bare_toolcall(text: str) -> tuple[str, dict] | None:
    """Pull (tool_name, args) out of a prose tool call so it can actually be
    run. Returns None if the text isn't a recognisable bare call."""
    if not looks_like_bare_toolcall(text):
        return None
    s = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip().strip("`").strip()
    m = re.match(r"^([a-z_]+)\s*\((.*)\)\s*$", s, re.IGNORECASE | re.DOTALL)
    if not m or m.group(1) not in _TOOL_NAMES:
        return None
    name, inside = m.group(1), m.group(2).strip()
    val = re.search(r"""(['"])(.*?)\1""", inside, re.DOTALL)
    if name == "surveillance_top_reported":
        num = re.search(r"\d+", inside)
        return name, {"limit": int(num.group(0))} if num else {}
    if not val:
        return None
    return name, {_PRIMARY_ARG.get(name, "query"): val.group(2).strip()}


_PREAMBLE_CUE = re.compile(
    r"^(?:okay[.,]?\s*|ok[.,]?\s*|alright[.,]?\s*|good[.,]?\s*|great[.,]?\s*)?"
    r"(now (?:i )?have|now that i have|i (?:now )?have (?:sufficient|enough|the)|"
    r"(?:i )?have (?:sufficient|enough|adequate) (?:evidence|information|grounding|context|detail)|"
    r"based on (?:my|the) (?:search|research|retriev|review|find)|"
    r"let me (?:now )?(?:write|draft|synthesi|put together|compile)|"
    r"i'?ll now (?:write|draft|synthesi|provide)|"
    r"(?:now )?writing (?:the |my )?(?:final )?(?:plan|answer|response)|"
    r"here(?:'?s| is) (?:the|my) (?:plan|answer|analysis|assessment)|"
    r"with (?:this|the) (?:evidence|information|context)|"
    r"having (?:gathered|retrieved|reviewed|collected)|"
    r"sufficient (?:grounding|evidence|information))",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    """Drop a leading process-narration sentence ("Now I have sufficient
    grounding…", "Based on my search…", "Have enough evidence. Writing plan
    now.") the model sometimes prepends before the real answer. Conservative:
    only short leading lines that aren't a heading, list item, or citation and
    match a known meta cue. Runs up to 3 times for multi-line preambles."""
    if not text:
        return text
    for _ in range(3):
        parts = text.lstrip().split("\n", 1)
        first = parts[0].strip()
        if (
            first
            and len(first) <= 200
            and not first.startswith(("#", "-", "*", ">", "|", "["))
            and _PREAMBLE_CUE.match(first)
        ):
            text = parts[1].lstrip() if len(parts) > 1 else ""
            if not text:
                break
        else:
            break
    return text


def _extract_api_error(response: httpx.Response) -> str:
    """Pull the human-readable message out of an Anthropic error response."""
    try:
        return str(response.json()["error"]["message"])[:300]
    except Exception:
        return response.text[:300] or "no detail"


class Retriever:
    """Runs retrieval functions, records a trace, and accumulates de-duplicated citations."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.trace: list[TraceStep] = []
        self.citations: list[Citation] = []
        self._seen: dict[str, int] = {}

    def _register(self, raw: dict) -> int:
        key = raw.get("url") or raw["title"]
        if key in self._seen:
            return self._seen[key]
        cid = len(self.citations) + 1
        series = None
        if raw.get("_series"):
            series = [{"period": p, "count": c} for p, c in raw["_series"]]
        detail = raw.get("_detail") or {}
        self.citations.append(Citation(
            id=cid,
            source=raw["source"],
            source_kind=raw["source_kind"],
            title=raw["title"],
            url=raw.get("url"),
            published=raw.get("published"),
            retrieved_at=_now_iso(),
            snippet=raw["snippet"],
            series=series,
            score=detail.get("similarity"),
            retrieval=detail.get("retrieval"),
        ))
        self._seen[key] = cid
        return cid

    async def run(self, tool_name: str, fn, params: dict, **kwargs) -> dict:
        started = time.perf_counter()
        status, note, source_api, raw_cites = "ok", None, "—", []
        try:
            raw_cites, source_api, note = await cached_call(
                tool_name, kwargs, lambda: fn(self.client, **kwargs)
            )
        except Exception as exc:  # network, JSON, timeout — recorded, never raised to the user
            status, note = "error", f"{type(exc).__name__}: {exc}"[:200]

        ids = [self._register(r) for r in raw_cites]
        self.trace.append(TraceStep(
            step=len(self.trace) + 1,
            tool=tool_name,
            source_api=source_api,
            params=params,
            result_count=len(raw_cites),
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=status,
            note=note,
        ))
        return {
            "citation_ids": ids,
            "results": [
                {
                    "cite": cid,
                    "title": self.citations[cid - 1].title,
                    "published": self.citations[cid - 1].published,
                    "url": self.citations[cid - 1].url,
                    "detail": raw.get("_detail") or self.citations[cid - 1].snippet,
                }
                for cid, raw in zip(ids, raw_cites)
            ],
            "note": note,
        }


async def _dispatch(retr: Retriever, name: str, args: dict) -> dict:
    if name == "search_local_guidelines":
        k = max(1, min(int(args.get("k", 4) or 4), 8))
        return await retr.run(name, search_knowledge_base, {"query": args.get("query", ""), "k": k},
                              query=args.get("query", ""), k=k)
    if name == "search_medical_literature":
        n = max(1, min(int(args.get("max_results", 6) or 6), 10))
        return await retr.run(name, search_literature, {"query": args.get("query", ""), "max_results": n},
                              query=args.get("query", ""), retmax=n)
    if name == "search_clinical_trials":
        n = max(1, min(int(args.get("max_results", 6) or 6), 20))
        return await retr.run(name, search_clinical_trials, {"query": args.get("query", ""), "max_results": n},
                              query=args.get("query", ""), page_size=n)
    if name == "search_drug_label":
        return await retr.run(name, search_drug_label, {"drug_name": args.get("drug_name", "")},
                              query=args.get("drug_name", ""))
    if name == "normalize_drug_name":
        return await retr.run(name, normalize_drug, {"drug_name": args.get("drug_name", "")},
                              drug_name=args.get("drug_name", ""))
    if name == "surveillance_signal":
        return await retr.run(name, surveillance_signal, {"term": args.get("term", "")},
                              term=args.get("term", ""))
    if name == "surveillance_top_reported":
        limit = max(1, min(int(args.get("limit", 15) or 15), 30))
        return await retr.run(name, surveillance_top_reported, {"limit": limit}, limit=limit)
    if name == "search_patient_education":
        return await retr.run(name, search_consumer_health, {"term": args.get("term", "")},
                              term=args.get("term", ""))
    if name == "search_prevention_guidance":
        return await retr.run(name, search_myhealthfinder, {"term": args.get("term", "")},
                              term=args.get("term", ""))
    if name == "search_nhs_conditions":
        return await retr.run(name, search_nhs_conditions, {"term": args.get("term", "")},
                              term=args.get("term", ""))
    if name == "medlineplus_connect_drug":
        return await retr.run(name, medlineplus_connect_drug, {"term": args.get("term", "")},
                              term=args.get("term", ""))
    return {"error": f"unknown tool: {name}"}


async def _call_claude(
    client: httpx.AsyncClient,
    messages: list[dict],
    *,
    use_tools: bool = True,
    system: str | None = None,
    tools: list | None = None,
    max_tokens: int | None = None,
) -> dict:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if ANTHROPIC_WORKSPACE_ID:
        headers["anthropic-workspace-id"] = ANTHROPIC_WORKSPACE_ID
    payload: dict = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens or MAX_TOKENS,
        "system": system or SYSTEM_PROMPT,
        "messages": messages,
    }
    if use_tools:
        payload["tools"] = tools or TOOLS
    resp = await with_retry(lambda: client.post(
        f"{ANTHROPIC_BASE_URL}/v1/messages",
        headers=headers,
        json=payload,
        timeout=60.0,
    ))
    resp.raise_for_status()
    return resp.json()


_COMPREHENSIVE_INSTRUCTION = (
    "\n\nRetrieve current evidence with your tools relevant to the clinician's question. "
    "Then write ONE answer using ONLY the sections below that are actually relevant and "
    "backed by something you retrieved, each as a `## ` heading, in this order when present:\n"
    "## Clinical decision support — options, tradeoffs, contraindications and boxed "
    "warnings for the question.\n"
    "## Treatment research — the evidence base and trial landscape (studies, phases, "
    "comparisons, gaps).\n"
    "## Disease surveillance — any adverse-event / safety-signal trend, read "
    "cautiously (spontaneous reports, no denominator, no causation).\n"
    "## Clinical narrative — a short plain-language background summary suitable for "
    "a note or briefing.\n"
    "Leave out a heading entirely if you have nothing relevant for it — do not include it "
    "just to say it doesn't apply, and do not pad. If the question asks a general "
    "'what's being reported' question with no specific drug, condition, or reaction named, "
    "use surveillance_top_reported rather than declining. Only if the question names no "
    "drug, condition, or topic at all and no tool has anything to offer, skip all four "
    "headings: say so in a sentence or two and ask what to clarify instead. Cite every "
    "clinical statement with [n]."
)

_TREATMENT_PLAN_INSTRUCTION = (
    "\n\nThis is a personalised treatment plan for the specific patient described in the "
    "case context above. Retrieve current evidence with your tools — drug label, dosing/"
    "renal-hepatic adjustment, and interaction information for anything in their current "
    "medication list is especially relevant — then write ONE plan.\n\n"
    "Structure it under `## ` headings that fit THIS case. Cover these three areas when "
    "there is something to say about them, but name the headings naturally (e.g. use the "
    "condition itself) and drop any area that genuinely doesn't apply — do not force an "
    "empty section or a fixed opening heading:\n"
    "1. the active problem(s) and evidence-based management options, with key tradeoffs and "
    "any contraindication specific to this patient (allergies, comorbidities, current meds);\n"
    "2. medication reconciliation — interactions, duplications, or contraindications against "
    "the patient's current medications and allergies, plus any dose adjustment their profile "
    "(renal/hepatic status, age) may require;\n"
    "3. follow-up and monitoring — what to check, how often, and red-flag findings that "
    "should trigger earlier review.\n"
    "This is decision support, not an order — phrase every recommendation as an option for "
    "the treating clinician to weigh. Cite every clinical statement with [n]. Get straight "
    "into the plan: no preamble, no notes about what the searches did or didn't return.\n\n"
    "BUT FIRST: if the case context is too thin to personalise anything — e.g. only "
    "demographics and no active problem, medication, presenting complaint, or relevant "
    "result — do NOT write a structured plan and do NOT list generic options. Instead "
    "reply with just a short 3-4 line note: state that there isn't enough information for "
    "a personalised plan, and name the specific items needed (e.g. active problem list, "
    "current medications, renal/hepatic function, presenting complaint). Nothing more."
)


async def _llm_plan(client: httpx.AsyncClient, retr: Retriever, req: QueryRequest) -> tuple[str, int]:
    prompt = f"Workflow: {MODE_LABEL[req.mode]}\n\nClinician query:\n{req.query}"
    if req.case_context:
        prompt += f"\n\nDe-identified case context:\n{req.case_context}"
    if req.mode == "comprehensive":
        prompt += _COMPREHENSIVE_INSTRUCTION
    elif req.mode == "treatment_plan":
        prompt += _TREATMENT_PLAN_INSTRUCTION
    else:
        prompt += ("\n\nRetrieve current evidence with your tools, then answer. "
                   "Cite every clinical statement with [n] using the citation ids from tool results.")

    # A comprehensive / treatment-plan answer has multiple sections and needs more
    # room than a single-workflow reply.
    answer_tokens = 4096 if req.mode in ("comprehensive", "treatment_plan") else MAX_TOKENS

    messages: list[dict] = [{"role": "user", "content": prompt}]
    rounds = 0
    recoveries = 0

    def _text_of(response: dict) -> str:
        return "\n".join(
            b["text"] for b in response.get("content", []) if b.get("type") == "text"
        ).strip()

    for _ in range(MAX_TOOL_ROUNDS):
        response = await _call_claude(client, messages, max_tokens=answer_tokens)
        content = response.get("content", []) or []
        stop = response.get("stop_reason")
        calls = [b for b in content if b.get("type") == "tool_use"]

        # A turn can carry tool calls even when it stopped on "max_tokens" rather
        # than "tool_use" — run them and keep going instead of bailing with an
        # empty answer.
        if calls and stop in ("tool_use", "max_tokens"):
            rounds += 1
            messages.append({"role": "assistant", "content": content})
            outs = await asyncio.gather(
                *(_dispatch(retr, b["name"], b.get("input") or {}) for b in calls)
            )
            tool_results = [
                {"type": "tool_result", "tool_use_id": b["id"], "content": json.dumps(o)[:9000]}
                for b, o in zip(calls, outs)
            ]
            messages.append({"role": "user", "content": tool_results})
            continue

        text = _text_of(response)
        if looks_like_bare_toolcall(text) and recoveries < 2:
            # The model wrote a tool call as prose instead of invoking it.
            # Push it to either call the tool for real or answer from what's
            # already retrieved.
            recoveries += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                "That is function-call syntax written as text, not a real tool call, "
                "and not a valid answer. Either invoke the tool through the tool "
                "interface to get what you need, or — if the evidence above is already "
                "enough — write the full plan now. Never reply with a bare tool call."
            )})
            continue

        stripped = strip_preamble(text)
        if stripped:
            return stripped, rounds
        # Empty, or nothing but a preamble line — fall through to force a real
        # answer from whatever evidence was gathered.
        if text:
            messages.append({"role": "assistant", "content": content})
        break

    # Round budget spent. Force a final answer from the evidence already gathered,
    # with tools withheld so the model cannot keep retrieving.
    final_nudge = ("Evidence gathering is complete. Write your final answer now from the "
                   "sources already retrieved above. Do not request more information. "
                   "Cite with [n]. If the evidence is thin, say so and answer with what there is.")
    if req.mode == "comprehensive":
        final_nudge += _COMPREHENSIVE_INSTRUCTION
    elif req.mode == "treatment_plan":
        final_nudge += _TREATMENT_PLAN_INSTRUCTION
    messages.append({"role": "user", "content": final_nudge})
    final = await _call_claude(client, messages, use_tools=False, max_tokens=answer_tokens)
    final_text = _text_of(final)
    if looks_like_bare_toolcall(final_text):
        messages.append({"role": "assistant", "content": final["content"]})
        messages.append({"role": "user", "content": (
            "Write the plan in prose now, from the sources already retrieved above. "
            "Do not output a tool call or function-call syntax."
        )})
        final = await _call_claude(client, messages, use_tools=False, max_tokens=answer_tokens)
        final_text = _text_of(final)

    final_text = strip_preamble(final_text)
    if final_text and not looks_like_bare_toolcall(final_text):
        return final_text, rounds

    # Last resort: the model will not produce prose. Run the search it kept
    # asking for plus the standard plan retrieval, and hand back an evidence-only
    # answer so the clinician never sees a raw tool call or an empty response.
    parsed = parse_bare_toolcall(final_text)
    if parsed:
        try:
            await _dispatch(retr, parsed[0], parsed[1])
        except Exception:
            pass
    if not retr.citations:
        await _deterministic_plan(retr, req)
    return _evidence_only_answer(req, retr, intro=(
        "**Automated synthesis was unavailable for this request** — the model "
        "repeatedly returned a search step instead of a written answer. The "
        "sources gathered are listed below; a clinician must synthesise them."
    )), rounds


async def _deterministic_plan(retr: Retriever, req: QueryRequest) -> None:
    """Run every source for the workflow concurrently — they are independent.

    "comprehensive" runs the union of all four workflows' sources.
    """
    m = req.mode
    _all = m == "comprehensive"
    term = primary_term(req.query)
    # External sources first, so local-knowledge citations get the trailing ids.
    tasks = [
        retr.run("search_medical_literature", search_literature,
                 {"query": req.query, "max_results": 6}, query=req.query, retmax=6),
    ]
    if _all or m == "surveillance":
        tasks.append(retr.run("surveillance_signal", surveillance_signal, {"term": term}, term=term))
    if _all or m == "treatment_research":
        tasks.append(retr.run("search_clinical_trials", search_clinical_trials,
                              {"query": req.query, "max_results": 6}, query=req.query, page_size=6))
    if _all or m in ("decision_support", "treatment_research", "treatment_plan"):
        tasks.append(retr.run("search_drug_label", search_drug_label, {"drug_name": term}, query=term))
    if _all or m in ("decision_support", "treatment_plan"):
        tasks.append(retr.run("normalize_drug_name", normalize_drug, {"drug_name": term}, drug_name=term))
        tasks.append(retr.run("medlineplus_connect_drug", medlineplus_connect_drug, {"term": term}, term=term))
    if _all or m == "narratives":
        tasks.append(retr.run("search_patient_education", search_consumer_health, {"term": term}, term=term))
        tasks.append(retr.run("search_prevention_guidance", search_myhealthfinder, {"term": term}, term=term))
        tasks.append(retr.run("search_nhs_conditions", search_nhs_conditions, {"term": term}, term=term))
    await asyncio.gather(*tasks)
    # Local knowledge base last.
    await retr.run("search_local_guidelines", search_knowledge_base,
                   {"query": req.query, "k": 4}, query=req.query, k=4)


def _evidence_only_answer(req: QueryRequest, retr: Retriever, intro: str | None = None) -> str:
    n = len(retr.citations)
    label = MODE_LABEL[req.mode].lower()
    lines = [
        intro or (
            "**Evidence-only mode.** No language model is configured "
            "(`ANTHROPIC_API_KEY` is unset), so the agent has retrieved and organised "
            "primary sources but has *not* written a clinical interpretation."
        ),
        "",
        f"**{n} source{'s' if n != 1 else ''}** retrieved for this {label} query. "
        "Each is listed in the Evidence panel with its origin API, retrieval time, "
        "and a direct link. A licensed clinician must synthesise them.",
    ]
    if n and req.mode == "comprehensive":
        # Group the retrieved sources under the four sub-workflow headings.
        buckets: dict[str, list[str]] = {
            "Clinical decision support": ["fda-label", "nomenclature"],
            "Treatment research": ["clinical-trial", "literature"],
            "Disease surveillance": ["pharmacovigilance"],
            "Clinical narrative": ["patient-education"],
        }
        seen: set[int] = set()
        for heading, kinds in buckets.items():
            group = [c for c in retr.citations if c.source_kind in kinds]
            if not group:
                continue  # nothing relevant retrieved — leave the heading out entirely
            lines += ["", f"## {heading}",
                      *[f"- [{c.id}] {c.title} — {c.source}" for c in group]]
            seen.update(c.id for c in group)
        other = [c for c in retr.citations if c.id not in seen]
        if other:
            lines += ["", "## Local guidance & other",
                      *[f"- [{c.id}] {c.title} — {c.source}" for c in other]]
    elif n:
        lines.append("")
        lines += [f"- [{c.id}] {c.title} — {c.source}" for c in retr.citations]
    else:
        lines += ["", "_No sources matched. Broaden the query or check terminology "
                  "(for drugs, try the generic name)._"]
    lines += ["", "Clinician review required before any patient-impacting action."]
    return "\n".join(lines)


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _local_last(citations: list[Citation], answer: str) -> tuple[list[Citation], str]:
    """Renumber so local-knowledge citations trail the external ones, rewriting the
    [n] markers in the answer to match. Stable within each group."""
    order = sorted(range(len(citations)),
                   key=lambda i: citations[i].source_kind == "local-knowledge")
    if order == list(range(len(citations))):
        return citations, answer
    remap = {citations[old].id: new for new, old in enumerate(order, start=1)}
    new_citations = [citations[old].model_copy(update={"id": new})
                     for new, old in enumerate(order, start=1)]

    def _sub(m: re.Match) -> str:
        nums = [remap.get(int(x), int(x)) for x in re.split(r"\s*,\s*", m.group(1))]
        return "[" + ", ".join(str(n) for n in nums) + "]"

    return new_citations, _CITE_RE.sub(_sub, answer)


async def attach_fhir_context(req: QueryRequest) -> None:
    """Fold a selected FHIR patient's summary into req.case_context, keeping any
    clinician-typed notes that arrived in case_context."""
    if not getattr(req, "fhir_patient_id", None):
        return
    manual = (req.case_context or "").strip()
    try:
        from .fhir import patient_context
        summary = (await patient_context(req.fhir_patient_id)).get("summary") or ""
    except Exception:  # sandbox server flaky / patient not found — proceed with notes only
        summary = ""
    if summary and manual:
        req.case_context = f"{summary}\n\nClinician-added presenting picture: {manual}"
    elif summary:
        req.case_context = summary
    # else: leave req.case_context as the clinician's notes (or None)


def missing_plan_context(req: QueryRequest) -> QueryResponse | None:
    """For treatment_plan mode with no case context, the short guidance response
    (shared by the built-in engine and the Tree-of-Thoughts engine so neither
    wastes model calls). Call *after* attach_fhir_context(). Returns None when
    there is enough to proceed.

    Two ways this happens, kept distinguishable: the browser sent no patient at
    all, vs. it sent one but the sandbox fetch came back empty/failed.
    """
    if req.mode != "treatment_plan" or (req.case_context or "").strip():
        return None
    if getattr(req, "fhir_patient_id", None):
        answer = (
            f"**Could not load patient {req.fhir_patient_id}'s chart just now.** "
            "The FHIR sandbox may be slow, unavailable, or this record may be empty, "
            "and no presenting details were typed to fall back on. Try again in a "
            "moment, or add presenting details in the notes box and generate the plan again."
        )
        reason = f"fhir_patient_id={req.fhir_patient_id} but no context/notes resolved for treatment_plan mode"
    else:
        answer = (
            "**No patient selected.** A personalised treatment plan is built from a "
            "patient's problem list, medications, and allergies. Select a patient from "
            "the Patient panel, or add presenting details, then generate the plan again."
        )
        reason = "No patient context supplied for treatment_plan mode"
    return QueryResponse(
        mode=req.mode, query=req.query, answer_markdown=answer,
        citations=[], retrieval_trace=[], llm=LLMInfo(used=False, reason=reason),
        human_review_required=True, generated_at=_now_iso(),
    )


async def run_query(req: QueryRequest) -> QueryResponse:
    await attach_fhir_context(req)
    guard = missing_plan_context(req)
    if guard is not None:
        return guard

    client = get_client()  # shared, connection-pooled; lives for the app's lifetime
    retr = Retriever(client)

    if ANTHROPIC_API_KEY:
        try:
            answer, rounds = await _llm_plan(client, retr, req)
            llm = LLMInfo(used=True, model=CLAUDE_MODEL, tool_rounds=rounds)
        except httpx.HTTPStatusError as exc:
            if not retr.trace:
                await _deterministic_plan(retr, req)
            detail = _extract_api_error(exc.response)
            answer = _evidence_only_answer(
                req, retr,
                intro=(f"**Language-model synthesis unavailable** "
                       f"(Anthropic API returned {exc.response.status_code}: {detail}). "
                       "The evidence below was still retrieved; a clinician must synthesise it."),
            )
            llm = LLMInfo(used=False, reason=f"LLM error {exc.response.status_code}: {detail}")
    else:
        await _deterministic_plan(retr, req)
        answer = _evidence_only_answer(req, retr)
        llm = LLMInfo(used=False, reason="ANTHROPIC_API_KEY not set — evidence-only retrieval mode")

    citations, answer = _local_last(retr.citations, answer)
    return QueryResponse(
        mode=req.mode,
        query=req.query,
        answer_markdown=answer,
        citations=citations,
        retrieval_trace=retr.trace,
        llm=llm,
        human_review_required=True,
        generated_at=_now_iso(),
    )
