"""FastAPI entrypoint: JSON API under /api, static clinical console at /."""
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .auth import COOKIE, check_credentials, cookie_max_age, issue_token, verify_token

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_WORKSPACE_ID,
    CLAUDE_MODEL,
    FHIR_BASE_URL,
    KNOWLEDGE_DIR,
)
from .agent import run_query
from .fhir import list_patients_with_data, patient_context, search_patients
from .crew import run_crew
from .lc_agent import LANGCHAIN_AVAILABLE, run_langchain_query
from .models import QueryRequest, QueryResponse, ThinkResponse
from .net import aclose_client, cache_stats, clear_cache
from .rag import get_index, reindex
from .rag_graph import RAG_GRAPH_AVAILABLE, run_rag_graph
from .tot import run_tot
from . import review_store

try:
    from monitoring.drift import log_query
except Exception:  # monitoring is optional
    def log_query(**_): ...

WEB_DIR = Path(__file__).resolve().parent.parent / "webui"

app = FastAPI(
    title="Clinical Agent",
    version="1.0.0",
    description="Evidence-grounded clinical decision support with full retrieval traceability.",
)

# --- auth gate: every /api/* route except login + health needs a valid session
_OPEN_API = {"/api/login", "/api/logout", "/api/health", "/api/me"}


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in _OPEN_API:
        if not verify_token(request.cookies.get(COOKIE)):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)


@app.post("/api/login")
async def login(payload: dict, response: Response) -> dict:
    role = check_credentials(str(payload.get("username", "")), str(payload.get("password", "")))
    if not role:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    response.set_cookie(
        COOKIE, issue_token(str(payload["username"]).strip(), role),
        max_age=cookie_max_age(), httponly=True, samesite="lax", path="/",
    )
    return {"ok": True, "role": role}


@app.post("/api/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request) -> dict:
    data = verify_token(request.cookies.get(COOKIE))
    if not data:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": data["u"], "role": data.get("r", "admin")}


def _require(request: Request, *roles: str) -> dict:
    """Return the session payload, or raise 401/403 if not the right role."""
    data = verify_token(request.cookies.get(COOKIE))
    if not data:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if roles and data.get("r", "admin") not in roles:
        raise HTTPException(status_code=403, detail=f"Requires role: {', '.join(roles)}")
    return data


@app.get("/api/health")
async def health() -> dict:
    idx = get_index()
    return {
        "status": "ok",
        "llm_configured": bool(ANTHROPIC_API_KEY),
        "workspace_id_set": bool(ANTHROPIC_WORKSPACE_ID),
        "model": CLAUDE_MODEL if ANTHROPIC_API_KEY else None,
        "mode": "synthesis" if ANTHROPIC_API_KEY else "evidence-only",
        "engines": {
            "builtin": True,
            "crew": bool(ANTHROPIC_API_KEY),
            "langchain": LANGCHAIN_AVAILABLE and bool(ANTHROPIC_API_KEY),
            "rag_graph": RAG_GRAPH_AVAILABLE and bool(ANTHROPIC_API_KEY),
            "tot": bool(ANTHROPIC_API_KEY),
        },
        "knowledge_base": {
            "documents": idx.n_docs,
            "chunks": idx.n_chunks,
            "retrieval": idx.mode,
            "embedding_model": idx.model,
            "built_at": idx.built_at,
            "dir": str(KNOWLEDGE_DIR),
        },
        "retrieval_cache": cache_stats(),
        "fhir_base": FHIR_BASE_URL,
    }


@app.get("/api/fhir/patients")
async def fhir_patients(name: str | None = None, count: int = 40, only_with_data: bool = True) -> dict:
    """List synthetic patients — the first `count` with no `name`, or a name
    search when `name` is given. `only_with_data` (default true) drops patients
    with an empty chart (no conditions/symptoms/meds/results/allergies) so a
    clinician never lands on a blank record."""
    try:
        patients = (await list_patients_with_data(name, count) if only_with_data
                    else await search_patients(name, count))
        return {"base": FHIR_BASE_URL, "patients": patients}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"FHIR server error: {exc}") from exc


@app.get("/api/fhir/patient/{patient_id}/context")
async def fhir_patient_context(patient_id: str) -> dict:
    """Problem list, medications, recent results, and allergies for one synthetic
    patient, plus a plain-text summary usable as query case context."""
    try:
        return await patient_context(patient_id)
    except httpx.HTTPStatusError as exc:
        code = 404 if exc.response.status_code == 404 else 502
        raise HTTPException(status_code=code, detail=f"FHIR: {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"FHIR server error: {exc}") from exc


@app.post("/api/cache/clear")
async def cache_clear() -> dict:
    """Drop the in-process retrieval response cache."""
    return {"cleared_entries": clear_cache()}


@app.on_event("shutdown")
async def _shutdown() -> None:
    await aclose_client()


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        resp = await run_query(req)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream data source error: {exc}") from exc
    except Exception as exc:  # transient upstream/model hiccup — 502, not a bare 500
        raise HTTPException(status_code=502, detail=f"Query failed: {type(exc).__name__}: {exc}") from exc
    log_query(engine="builtin", req=req, resp=resp,
              latency_ms=int((time.perf_counter() - started) * 1000))
    return resp


@app.post("/api/rag/reindex")
async def rag_reindex() -> dict:
    """Rebuild the local knowledge-base index from files in the knowledge/ directory."""
    return reindex()


_KB_SUFFIXES = {".md", ".markdown", ".txt", ".text"}


@app.get("/api/kb/doc/{doc_path:path}", response_class=PlainTextResponse)
async def kb_doc(doc_path: str) -> str:
    """Serve a raw local knowledge-base document so its citations can link to it."""
    base = KNOWLEDGE_DIR.resolve()
    target = (base / unquote(doc_path)).resolve()
    if not (target == base or base in target.parents):
        raise HTTPException(status_code=404, detail="Document not found.")
    if not target.is_file() or target.suffix.lower() not in _KB_SUFFIXES:
        raise HTTPException(status_code=404, detail="Document not found.")
    return target.read_text(encoding="utf-8", errors="replace")


@app.post("/api/query/langchain", response_model=QueryResponse)
async def query_langchain(req: QueryRequest) -> QueryResponse:
    """Same as /api/query, but evidence collection runs through a LangChain /
    LangGraph ReAct agent instead of the built-in loop."""
    if not LANGCHAIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LangChain engine not installed. pip install -r requirements-langchain.txt",
        )
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="LangChain engine requires ANTHROPIC_API_KEY.")
    try:
        return await run_langchain_query(req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({exc.response.status_code}): {exc.response.text[:300]}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc


@app.post("/api/query/rag-graph", response_model=QueryResponse)
async def query_rag_graph(req: QueryRequest) -> QueryResponse:
    """Corrective-RAG pipeline as an explicit LangGraph StateGraph:
    retrieve → grade → (rewrite → retrieve)? → generate."""
    if not RAG_GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LangGraph RAG engine not installed. pip install -r requirements-langchain.txt",
        )
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="LangGraph RAG engine requires ANTHROPIC_API_KEY.")
    try:
        return await run_rag_graph(req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({exc.response.status_code}): {exc.response.text[:300]}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc


@app.post("/api/query/tot", response_model=QueryResponse)
async def query_tot(req: QueryRequest) -> QueryResponse:
    """Tree-of-Thoughts: propose several candidate approaches, retrieve + draft
    each, score against evidence, deepen the strongest, synthesise one answer."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Tree-of-Thoughts engine requires ANTHROPIC_API_KEY.")
    try:
        return await run_tot(req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({exc.response.status_code}): {exc.response.text[:300]}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc


@app.post("/api/think", response_model=ThinkResponse)
async def think(req: QueryRequest) -> ThinkResponse:
    """Thought generator: a Planner → Researcher → Synthesist → Safety Reviewer
    crew, returning the full chain of thoughts alongside the cited answer."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Thought generator requires ANTHROPIC_API_KEY (every crew agent is LLM-based).",
        )
    try:
        return await run_crew(req)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({exc.response.status_code}): {exc.response.text[:300]}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc


# --- treatment-plan review workflow -----------------------------------------

@app.post("/api/review/plans")
async def review_plan_create(payload: dict, request: Request) -> dict:
    """Admin sends a generated treatment plan to the doctor queue for approval."""
    sess = _require(request, "admin")
    answer = str(payload.get("answer_markdown", "")).strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer_markdown is required.")
    row = review_store.add_plan(
        query=str(payload.get("query", "")),
        answer_markdown=answer,
        citations=payload.get("citations") or [],
        created_by=sess["u"],
        fhir_patient_id=payload.get("fhir_patient_id"),
        patient_label=(str(payload.get("patient_label")).strip()[:120]
                       if payload.get("patient_label") else None),
    )
    return {"id": row["id"], "status": row["status"]}


@app.get("/api/review/plans")
async def review_plans(request: Request, scope: str = "queue") -> dict:
    """Doctors see only what's awaiting their review. Admins default to the same
    action queue (plans doctors approved) but with `?scope=all` get every plan
    they've sent, in any status, for tracking."""
    sess = _require(request, "admin", "doctor")
    if sess["r"] == "doctor":
        statuses = review_store.DOCTOR_QUEUE
    elif scope == "all":
        statuses = None  # every status
    else:
        statuses = review_store.ADMIN_QUEUE
    return {"role": sess["r"], "plans": review_store.list_plans(statuses=statuses)}


@app.get("/api/review/plans/{plan_id}")
async def review_plan_detail(plan_id: str, request: Request) -> dict:
    _require(request, "admin", "doctor")
    plan = review_store.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found.")
    return plan


@app.post("/api/review/plans/{plan_id}/review")
async def review_plan_doctor(plan_id: str, payload: dict, request: Request) -> dict:
    """Doctor decision: approve, or request changes, with optional review notes."""
    sess = _require(request, "doctor")
    decision = str(payload.get("decision", "")).strip()
    if decision not in ("approve", "request_changes", "reject"):
        raise HTTPException(status_code=400,
                            detail="decision must be 'approve', 'request_changes', or 'reject'.")
    updated = review_store.doctor_review(
        plan_id, notes=str(payload.get("notes", "")), decision=decision, reviewer=sess["u"],
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Plan not found.")
    return updated


@app.post("/api/review/plans/{plan_id}/comment")
async def review_plan_comment(plan_id: str, payload: dict, request: Request) -> dict:
    """Append a comment to a plan. Admin-only — the doctor replies through the
    review action, not here, and can only read these."""
    sess = _require(request, "admin")
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment text is required.")
    updated = review_store.add_comment(plan_id, text=text, by=sess["u"], role=sess["r"])
    if not updated:
        raise HTTPException(status_code=404, detail="Plan not found.")
    return updated


@app.post("/api/review/plans/{plan_id}/comment/{comment_id}")
async def review_plan_comment_edit(plan_id: str, comment_id: str, payload: dict, request: Request) -> dict:
    """Update one of your own comments. Admin-only."""
    sess = _require(request, "admin")
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment text is required.")
    updated = review_store.edit_comment(plan_id, comment_id, text=text, by=sess["u"])
    if updated is None:
        raise HTTPException(status_code=404, detail="Plan not found.")
    if updated.get("_error"):
        raise HTTPException(status_code=403, detail=updated["_error"])
    return updated


@app.get("/api/review/my-reviews")
async def review_my_reviews(request: Request) -> dict:
    """Reviews this doctor completed in the last 24 hours."""
    sess = _require(request, "doctor")
    return {"reviews": review_store.doctor_recent_reviews(sess["u"], hours=24)}


@app.post("/api/review/plans/{plan_id}/respond")
async def review_plan_respond(plan_id: str, payload: dict, request: Request) -> dict:
    """Admin replies to a doctor's change request and sends the plan back for review."""
    sess = _require(request, "admin")
    comment = str(payload.get("comment", "")).strip()
    if not comment:
        raise HTTPException(status_code=400, detail="A comment is required to send the plan back.")
    updated = review_store.admin_respond(plan_id, comment=comment, admin=sess["u"])
    if updated is None:
        raise HTTPException(status_code=404, detail="Plan not found.")
    if updated.get("_error"):
        raise HTTPException(status_code=409, detail=updated["_error"])
    return updated


@app.post("/api/review/plans/{plan_id}/acknowledge")
async def review_plan_ack(plan_id: str, payload: dict, request: Request) -> dict:
    """Admin's final review of a doctor-approved plan."""
    sess = _require(request, "admin")
    updated = review_store.admin_acknowledge(
        plan_id, note=str(payload.get("note", "")), admin=sess["u"],
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Plan not found.")
    if updated.get("_error"):
        raise HTTPException(status_code=409, detail=updated["_error"])
    return updated


_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}


@app.get("/")
async def index(request: Request) -> FileResponse:
    """Routed by session: doctor → review portal, admin → console, else login.
    Always no-cache so a stale copy never pins old JS/CSS or the wrong page."""
    sess = verify_token(request.cookies.get(COOKIE))
    if not sess:
        page = "login.html"
    elif sess.get("r") == "doctor":
        page = "doctor.html"
    else:
        page = "index.html"
    return FileResponse(WEB_DIR / page, headers=_NO_CACHE)


@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(WEB_DIR / "login.html", headers=_NO_CACHE)


@app.get("/doctor")
async def doctor_page(request: Request) -> FileResponse:
    """Separate doctor sign-in; once signed in as a doctor, '/' serves the portal."""
    sess = verify_token(request.cookies.get(COOKIE))
    page = "doctor.html" if sess and sess.get("r") == "doctor" else "doctor-login.html"
    return FileResponse(WEB_DIR / page, headers=_NO_CACHE)


@app.get("/approvals")
async def approvals_page(request: Request) -> FileResponse:
    """Admin-only: review the plans doctors have approved."""
    sess = verify_token(request.cookies.get(COOKIE))
    if not sess or sess.get("r") != "admin":
        return FileResponse(WEB_DIR / "login.html", headers=_NO_CACHE)
    return FileResponse(WEB_DIR / "approvals.html", headers=_NO_CACHE)


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
