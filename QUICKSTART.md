# QUICKSTART — running & reviewing ClinicalAgent

Evidence-grounded clinical decision support: a FastAPI service + vanilla-JS console
that answers clinical questions by **retrieving current medical information at
request time** (FDA labelling, PubMed, RxNorm, FAERS, MedlinePlus, a local RAG
corpus) and synthesising a cited answer. A clinician-review gate stands between
every output and any patient-impacting use.

> Reference tool for licensed professionals. Not a diagnostic device. Human review
> is required before any patient-impacting decision.

---

## 1. Run it from a fresh clone

### Prerequisites
- **Python 3.14** (3.11+ works; `langchain-community` extras need a numpy wheel, which 3.14 lacks — the core app does not use them)
- **git**
- Optional: an **Anthropic API key** for LLM synthesis. Without it the app still runs in **evidence-only mode** (retrieval + provenance, no interpretation) — so it starts and serves with **zero configuration**.

### What the repo does and does not ship
| Ships in the clone | Built on first run / not in the clone |
| --- | --- |
| all source, the `knowledge/` RAG corpus (~1,085 docs), `data/qa.jsonl` + `data/sft_corpus.jsonl` | `.venv/` — `run.ps1` creates it |
| `.env.example` | `.env` — you copy it (or skip for evidence-only mode) |
| | `knowledge/.kb_index.json` — the RAG index; **auto-builds from `knowledge/` on first query** (~a few seconds, TF-IDF). Force it with `python -m api.rag`. |
| | `data/treatment_plans.json` — approval state; created on the first "Send to doctor" |

### Windows (PowerShell)
```powershell
git clone https://github.com/bhavanavankhede29/ClinicalAgent
cd ClinicalAgent

# optional: configure. Skip entirely to run in evidence-only mode.
Copy-Item .env.example .env
#   edit .env — set ANTHROPIC_API_KEY (+ ANTHROPIC_WORKSPACE_ID if the key is
#   identity-linked). Set a fixed SESSION_SECRET so logins survive a restart.

# run — creates .venv, installs requirements.txt, starts uvicorn with --reload
.\run.ps1
```

### macOS / Linux (no `run.ps1`)
```bash
git clone https://github.com/bhavanavankhede29/ClinicalAgent
cd ClinicalAgent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # optional; edit as above
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Verify the clone is working
```
GET http://127.0.0.1:8000/api/health
```
Expect `status: "ok"`, `knowledge_base.chunks` in the thousands (index built), and
`mode` = `synthesis` if you set a key, else `evidence-only`. Then open
**http://127.0.0.1:8000** and sign in.

| Login | Username / password | Where | Does |
| --- | --- | --- | --- |
| Admin | `admin` / `admin123` | `/` | the console — ask questions, generate treatment plans, send plans for approval, finalise approved plans at `/approvals` |
| Doctor | `doctor` / `admindoc` | `/doctor` | review portal — read pending plans + sources, add notes, **Approve / Request changes / Reject** |

> Use `http://127.0.0.1:8000`, **not** `localhost` — the session cookie is pinned to one origin.

### Keep it running
`run.ps1` starts a foreground server. Leave that terminal window open — if it
closes, the app stops (and the FHIR panel / queries start failing). For unattended
use, run it as a Windows Scheduled Task at logon, or `nssm`/`sc` as a service.

---

## 2. Use it

### Ask a clinical question
Type in the box, press **Enter** or click a button:

| Button | Sends mode | You get |
| --- | --- | --- |
| **Generate treatment plan** | `treatment_plan` | a patient-specific plan: active problems + options, medication reconciliation, follow-up & monitoring. Needs a patient (see below). Enter key triggers this. |
| **Detailed References** | `comprehensive` | an evidence synthesis across the relevant sections (decision support / trial landscape / safety signals / plain-language background). No patient required. |
| **Deep reasoning** (checkbox) | routes either button through `/api/query/tot` | Tree-of-Thoughts: proposes 3 approaches, scores each against the evidence, deepens the best, synthesises. Slower (~45s, 7–9 model calls). |

### Attach a patient (for treatment plans)
Left panel, two mutually-exclusive modes:
- **FHIR sandbox** — search/select a **synthetic** patient from `hapi.fhir.org`. Each row shows what the chart holds (`2 cond · 2 meds · 15 results`). Pick one with conditions **and** meds — a thin chart yields a short "not enough information" note by design.
- **Manual entry** — type age / sex / complaint / meds / allergies / history. Sends only `case_context`, no FHIR id, so nothing from a previously selected patient leaks in. The form is read live — "Apply patient" is just a preview.

### Treatment-plan approval workflow
1. Generate a real plan → click **Send to doctor for approval**.
2. Doctor signs in at `/doctor`, opens the plan, adds notes, **Approve / Request changes / Reject**.
3. `changes_requested` → admin replies on `/approvals` and **sends it back** (returns to `pending_review`).
4. `approved` → admin **Acknowledge & finalize**.

Every step is appended to the plan's `history[]` and shown as a thread on both
pages. Comments are separate from the status flow; the doctor portal also lists
that doctor's reviews from the last 24 h. State lives in one JSON file
(`data/treatment_plans.json`, git-ignored).

### Provenance on every answer
- `citations[]` — each source with its origin API, retrieval timestamp, direct link, snippet.
- `retrieval_trace[]` — every step: API called, params, result count, latency, status (including "no results" / errors).
- `llm` — whether a model synthesised, which one, how many tool rounds.
- `human_review_required` — always `true`. "Copy for record" is disabled until the reviewer ticks the gate.

---

## 3. Review the code

### Reviewer's fast path
```bash
git clone https://github.com/bhavanavankhede29/ClinicalAgent && cd ClinicalAgent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# read the orchestration + safety rules first
less api/agent.py            # SYSTEM_PROMPT, TOOLS, _llm_plan, answer guards

# NO KEY — exercise retrieval + RAG deterministically
.venv/bin/python -m api.rag                      # build + report the local index
.venv/bin/python -m uvicorn api.main:app --port 8000     # runs in evidence-only mode
#   → POST /api/query returns real citations + retrieval_trace, no synthesis

# WITH ANTHROPIC_API_KEY set — see the full reasoning trace on the CLI, no web server
.venv/bin/python -m api.crew "metformin use with reduced kidney function" --mode decision_support
```
Every retrieval function in `api/retrieval.py` is independently callable; the
crew CLI (`api/crew.py`) prints the Planner → Researcher → Synthesist → Reviewer
chain of thoughts plus the cited answer.

### Where to look first
| File | What it is |
| --- | --- |
| [`api/main.py`](api/main.py) | FastAPI app — all `/api/*` routes, auth guard, static console, page routing by role |
| [`api/agent.py`](api/agent.py) | core orchestration — the Claude tool-use loop (`_llm_plan`), the deterministic fallback, `SYSTEM_PROMPT`, the tool schemas, the answer guards |
| [`api/retrieval.py`](api/retrieval.py) | one async function per live source (openFDA, PubMed, ClinicalTrials.gov, RxNorm, FAERS, MedlinePlus, MyHealthfinder, NHS) |
| [`api/rag.py`](api/rag.py) | local knowledge base — chunk (500/100) + embed (Voyage or TF-IDF) + search over `knowledge/` |
| [`api/tot.py`](api/tot.py) | Tree-of-Thoughts engine (Deep reasoning) |
| [`api/fhir.py`](api/fhir.py) | sandbox FHIR R4 client — synthetic patient → problem list / meds / results as `case_context` |
| [`api/review_store.py`](api/review_store.py) | file-backed treatment-plan approval state machine |
| [`api/auth.py`](api/auth.py) | HMAC-signed cookie sessions, roles `admin` / `doctor` |
| [`api/net.py`](api/net.py) | shared pooled HTTP client, TTL response cache, SSL/transport retry |
| [`webui/`](webui/) | the console — `index.html`, `app.js`, `styles.css`; plus `doctor.html`, `approvals.html`, login pages. No build step. |
| [`mcp_server.py`](mcp_server.py) | the same retrieval layer exposed over Model Context Protocol |

### Reasoning engines (same `QueryResponse` shape, engine-independent provenance)
| Endpoint | Engine |
| --- | --- |
| `POST /api/query` | built-in Claude tool-use loop (≤6 rounds), or deterministic retrieval if no key |
| `POST /api/query/tot` | Tree-of-Thoughts |
| `POST /api/query/langchain` | LangGraph ReAct (needs `requirements-langchain.txt`) |
| `POST /api/query/rag-graph` | LangGraph corrective-RAG StateGraph |
| `POST /api/think` | 4-agent thought generator (`api/crew.py`) — returns a `thoughts[]` trace |

### Anti-hallucination / safety model
- With a key: `SYSTEM_PROMPT` requires every clinical statement to be grounded in a source retrieved *this turn*; the model must say so plainly when tools return nothing. No tool-name narration, no preamble; critical items (contraindications, boxed warnings) prefixed `⚠`.
- Without a key: **no synthesis at all** — retrieval + provenance only.
- Answer guards in `_llm_plan` / `run_tot`: a reply that is a bare tool call or empty is caught, the model is re-prompted, and failing that the request falls back to an organised evidence list — the raw tool call never reaches the UI.
- FHIR is synthetic-only (`FHIR_BASE_URL` must never point at real PHI).

### Health check
```
GET /api/health
```
→ `llm_configured`, `mode` (`synthesis` | `evidence-only`), `knowledge_base` (doc/chunk counts, `retrieval` = `voyage` | `tfidf`), `engines` availability, `fhir_base`.

### Run a query from the shell (after logging in for a cookie)
```powershell
$s = New-Object Microsoft.PowerShell.Commands.WebRequestSession
Invoke-RestMethod http://127.0.0.1:8000/api/login -Method Post -WebSession $s `
  -ContentType application/json -Body (@{username='admin';password='admin123'} | ConvertTo-Json) | Out-Null
Invoke-RestMethod http://127.0.0.1:8000/api/query -Method Post -WebSession $s `
  -ContentType application/json -Body (@{
    mode='treatment_plan'
    query='Personalized treatment plan.'
    case_context='58F. Type 2 diabetes; NAFLD, ALT 96. Metformin 1000 mg BID; atorvastatin 20 mg. NKDA.'
  } | ConvertTo-Json)
```

---

## 4. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "FHIR request failed" / every request errors | The server isn't running. Start it with `.\run.ps1` and keep the window open. |
| Signed in, bounced back to login | Cookie not kept — use `http://127.0.0.1:8000`, not `localhost`. Set a fixed `SESSION_SECRET` in `.env` so a restart doesn't invalidate sessions. |
| UI changes don't show | Browser cached `app.js`/`styles.css`. Hard-refresh (Ctrl+F5); the `?v=` query bump forces a refetch. |
| "There isn't enough information … for a personalised plan" | Working as designed — the selected patient's chart is thin. Pick a patient with conditions + meds, or use Manual entry. |
| Answer says "evidence-only" | `ANTHROPIC_API_KEY` not set in `.env` (the `run.ps1` banner reads the *shell* env; the app reads `.env`). |
| 400 `anthropic-workspace-id is required` | Your key is identity-linked — set `ANTHROPIC_WORKSPACE_ID` in `.env`. |
| RAG retrieval is `tfidf` not `voyage` | No `VOYAGE_API_KEY`, or the free tier returned 429 (auto-fallback). Fine for local use. |

Rebuild the RAG index after changing `knowledge/`: `python -m api.rag` or `POST /api/rag/reindex`.
