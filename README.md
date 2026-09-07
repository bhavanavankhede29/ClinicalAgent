# ClinicalAgent

Evidence-grounded clinical decision support. The agent answers clinical questions
by **retrieving current medical information at request time** — FDA labelling,
PubMed literature, RxNorm nomenclature, and FAERS pharmacovigilance data — rather
than relying on a model's pretrained knowledge. Every answer ships with the full
retrieval trace and a citation for each source, and a clinician-review gate stands
between the output and any patient-impacting use.

> Reference tool for licensed healthcare professionals. Not a diagnostic device.
> Human review is required before any patient-impacting decision.

## FastAPI app (`api/` + `webui/`)

The main interface: a three-pane clinical console backed by a FastAPI service.

| Workflow | What it retrieves |
| --- | --- |
| Clinical decision support | FDA label (indications, dosing, contraindications, interactions) + RxNorm concept + literature |
| Treatment research | PubMed literature + labelled indications for the therapy area |
| Disease surveillance | FAERS adverse-event report volume over time (a safety signal, charted) + literature |
| Clinical narrative | Sourced background for a summary, case report, or briefing |

### Run

```powershell
# optional — enables LLM synthesis of the retrieved evidence; without it the
# app runs in "evidence-only" mode (retrieval + provenance, no interpretation)
$env:ANTHROPIC_API_KEY = 'your-api-key'

.\run.ps1
```

Then open <http://127.0.0.1:8000>. `run.ps1` creates `.venv`, installs
`requirements.txt`, and starts `uvicorn api.main:app --reload`.

### How traceability works

`POST /api/query` returns, alongside the answer:

- `citations[]` — each source with its origin API, retrieval timestamp, direct
  link, and a snippet. Surveillance signals also carry a `series` for charting.
- `retrieval_trace[]` — every retrieval step: the API called, its parameters,
  result count, latency, and status (including "no results" and errors).
- `llm` — whether a model synthesised the answer, which model, and how many
  tool-use rounds it took.
- `human_review_required` — always `true`.

In the UI, inline `[n]` markers in the answer are buttons that jump to the
matching evidence card. "Copy for record" is disabled until the reviewer ticks
the review gate, and the copied block records whether review was confirmed.

### RAG — local knowledge base

Live APIs cover published evidence; the RAG layer covers what an institution
holds that is on no public API — local protocols, formulary rules, order-set
rationale, guideline adaptations.

- Put documents in [`knowledge/`](knowledge/) — `.pdf`, `.md`, `.txt`, `.html` —
  under the category folders (`guidelines/`, `protocols/`, `drug-references/`,
  `reviews/`, `local/`), then rebuild the index: `python -m api.rag` (or
  `POST /api/rag/reindex`).
- **Provenance** — each document carries `source` / `title` / `published` /
  `version` / `jurisdiction` / `url` via a `<name>.meta.json` sidecar or a
  `--- key: value ---` front-matter block. Retrieved passages are cited as
  `Local KB · <title> › <heading>  (source · version · date · jurisdiction)` and
  link to the source `url` when given.
- Load one document with provenance and reindex in a step:
  `python -m api.kb_load <file> --category guidelines --source "IDSA" --title "…" --published 2019-10 --version 2019 --jurisdiction US --url "…"`
- **Chunking:** 500-character sliding windows with 100 characters of overlap,
  never crossing a Markdown heading (`RAG_CHUNK_CHARS` / `RAG_CHUNK_OVERLAP`).
- **Embeddings:** with `VOYAGE_API_KEY` set, every chunk and every query is
  embedded with Voyage AI (`CLINICAL_AGENT_EMBED_MODEL`, default `voyage-3.5`)
  and retrieval is dense cosine similarity. Without a key it falls back to a
  local TF-IDF vector (pure Python, offline). The backend swap is contained in
  [`api/rag.py`](api/rag.py) — the agent interface is unchanged.
- The agent calls `search_local_guidelines` **first** on every query; matches are
  cited as `Local KB · <file> › <heading>` with a similarity score, and the
  system prompt tells the model that local policy overrides general sources when
  it applies.
- `GET /api/health` → `knowledge_base` reports document / chunk counts, the
  active `retrieval` mode (`voyage` | `tfidf`), and the `embedding_model`. The
  index auto-rebuilds when the mode or model no longer matches config.

### LangChain engine (`api/lc_agent.py`)

An alternative evidence-collection loop: the same retrieval + RAG functions
wrapped as LangChain tools, driven by a LangGraph ReAct agent
(`create_react_agent`) over `ChatAnthropic`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-langchain.txt
```

Then `POST /api/query/langchain` — identical request and `QueryResponse` shape as
`/api/query` (the shared `Retriever` still produces the citations and trace, so
provenance is engine-independent). `llm.model` is tagged `… (langchain/langgraph)`.
`GET /api/health` → `engines.langchain` shows whether it is available.

Pure Python, installs on 3.14. `langchain-community` does **not** — it pulls
`numpy`, which has no 3.14 wheel and no compiler on this box. The identity-linked
key's `anthropic-workspace-id` is passed via `ChatAnthropic(default_headers=…)`.

### LangGraph corrective-RAG (`api/rag_graph.py`)

`POST /api/query/rag-graph` runs an explicit **LangGraph `StateGraph`** rather
than a ReAct loop:

```
START → retrieve → grade ──sufficient──▶ generate → END
                     └── insufficient ──▶ rewrite ──▶ (retrieve again, once)
```

* **retrieve** — the workflow's deterministic source plan (live APIs + the local
  knowledge base / Voyage vector search) into one shared `Retriever`
* **grade** — an LLM check on whether the context can answer the question
* **rewrite** — if not, the LLM rewrites the search query and retrieval reruns once
* **generate** — the LLM writes the `[n]`-cited answer

Same `QueryResponse` shape and provenance as `/api/query`; `llm.model` is tagged
`… (langgraph corrective-RAG)` and `llm.reason` reports the grade and rewrite
count. Needs the same `requirements-langchain.txt` extras; `engines.rag_graph` in
`/api/health` shows availability.

### Tree-of-Thoughts (`api/tot.py`)

`POST /api/query/tot` explores several candidate *approaches* to the question in
parallel instead of committing to one line of reasoning:

```
propose 3 approaches
  └─ each: targeted retrieval → draft partial answer   (drafts run concurrently)
       └─ evaluate all drafts 0-10 (grounded in the cited evidence?)
            └─ keep top 1-2  →  deepen each (close the reviewer's biggest gap)
                 └─ synthesise one [n]-cited answer from the survivors
```

Breadth 3, depth 2, keep top 2 (top 1 if the runner-up trails by >3 points).
~7-9 LLM calls. Reuses the same shared `Retriever`, so `citations` and
`retrieval_trace` are pooled across every branch. Same `QueryResponse` shape;
`llm.model` is tagged `… (tree-of-thoughts)` and `llm.reason` records the per-
approach scores and which survived, e.g.
`3 approaches [#2:7, #0:6, #1:4] → kept #2(7), #0(6) → synthesised from 6 sources`.
Needs only `ANTHROPIC_API_KEY`; `engines.tot` in `/api/health` shows availability.

In the console, the **Deep reasoning** checkbox next to the buttons routes both
"Detailed References" and "Generate treatment plan" through this endpoint
instead of `/api/query` (choice is remembered in `localStorage`). The
`treatment_plan` "no patient context" guard is shared, so an empty deep-reasoning
plan request returns instantly instead of burning the tree.

### Thought generator (`api/crew.py`)

A multi-agent pass that shows its reasoning. `POST /api/think`
(same request body as `/api/query`) runs four agents in sequence and returns a
`thoughts[]` trace alongside the cited answer:

| Agent | Does |
| --- | --- |
| Retrieval Planner | picks which tools to run and writes the search terms (emits a plan + per-step rationale) |
| Evidence Researcher | executes the plan over the real retrieval + RAG tools; every call and its result is a thought |
| Clinical Synthesist | writes the `[n]`-cited Markdown answer from what was retrieved |
| Safety Reviewer | audits the draft against the safety rules; returns a verdict and drives one revision loop |

Each `thoughts[]` item is `{step, agent, kind, content, data}` where `kind` is
`plan | thought | action | observation | draft | critique | decision`. The
response also carries `citations`, `retrieval_trace`, `revisions`, and
`human_review_required`.

Requires `ANTHROPIC_API_KEY` (every agent is an LLM call). CLI:

```powershell
.\.venv\Scripts\python.exe -m api.crew "metformin use with reduced kidney function" --mode decision_support
```

`api/crew.py` is self-contained — role/goal/backstory agents, sequential tasks,
and a step callback that emits each thought. It depends on no agent framework.

### FHIR patient context (`api/fhir.py`)

Optional grounding in a **synthetic** patient from a sandbox FHIR R4 server
(`FHIR_BASE_URL`, default `https://hapi.fhir.org/baseR4` — never a server with
real PHI).

| Endpoint | Returns |
| --- | --- |
| `GET /api/fhir/patients?name=` | matching synthetic patients (id, name, gender, DOB) |
| `GET /api/fhir/patient/{id}/context` | problem list, active meds, recent results, allergies + a plain-text `summary` |

Any query (`/api/query`, `/api/query/langchain`, `/api/query/rag-graph`,
`/api/query/tot`, `/api/think`) may include `"fhir_patient_id": "<id>"` — the patient's summary is
pulled from `Condition` / `MedicationRequest` / `Observation` /
`AllergyIntolerance` and folded into `case_context` before retrieval and
synthesis. `GET /api/health` reports `fhir_base`.

The console's Patient panel has two modes: **FHIR sandbox** (search + select a
synthetic record) and **Manual entry** (type age / sex / complaint / meds /
allergies / history by hand). Manual entry sends only `case_context` with **no
`fhir_patient_id`**, so a previously selected FHIR patient can't leak into the
request — the two modes are mutually exclusive and switching clears the other.

### Treatment-plan approval workflow (`api/review_store.py`)

A treatment plan is filed for review only when the admin clicks **Send to
doctor for approval** on the generated plan (real plan — not the "insufficient
info" note).

```
generated → [admin sends] → pending_review → doctor approves → approved → admin acknowledges → acknowledged
                                 ▲        ├ doctor requests changes → changes_requested
                                 │        └ doctor rejects → rejected (terminal)
                                 └────── admin comments & resends ──────┘
```

A `changes_requested` plan isn't a dead end: on `/approvals` the admin sees the
doctor's requested changes, adds a reply, and **Send back to doctor** — status
returns to `pending_review` with the admin's comment attached, and the doctor
re-reviews. Every step (submit, request changes, admin response, approve,
reject, acknowledge) is appended to the plan's `history[]` and shown as a
thread on both the doctor and admin pages.

**Comments** — separate from the status flow, either role can post free-text
comments on a plan (and edit their own); both pages render the thread live so
the admin can revise a note and the doctor sees the current text. **The doctor
portal also shows every review that doctor completed in the last 24 hours**
(`GET /api/review/my-reviews`) in a panel below the queue.

Two logins (`ADMIN_USER`/`ADMIN_PASSWORD`, `DOCTOR_USER`/`DOCTOR_PASSWORD`;
defaults `admin`/`admin123` and `doctor`/`admindoc`). The session cookie carries
the role.

| Page | Who | Does |
| --- | --- | --- |
| `/` | admin | the console; topbar **Approvals** link → `/approvals` |
| `/doctor` | doctor | separate green-themed sign-in → the review portal |
| review portal (`/` as doctor) | doctor | read each pending plan + its sources, add review notes, **Approve** or **Request changes** |
| `/approvals` | admin | read doctor-approved plans with the doctor's notes, add a final note, **Acknowledge & finalize** |

| Endpoint | Role | |
| --- | --- | --- |
| `POST /api/review/plans` | admin | send a generated plan `{query, answer_markdown, citations, fhir_patient_id, patient_label}` to the doctor queue |
| `GET /api/review/plans` | admin, doctor | role-scoped queue (doctor: pending/changes; admin: approved/acknowledged) |
| `GET /api/review/plans/{id}` | admin, doctor | full plan record |
| `POST /api/review/plans/{id}/review` | doctor | `{decision: "approve"\|"request_changes"\|"reject", notes}` (`reject` is terminal) |
| `POST /api/review/plans/{id}/respond` | admin | `{comment}` — reply to a change request; sends the plan back to `pending_review` (409 unless `changes_requested`) |
| `POST /api/review/plans/{id}/acknowledge` | admin | `{note}` — only on an `approved` plan (409 otherwise) |
| `POST /api/review/plans/{id}/comment` | admin, doctor | `{text}` — append a discussion comment |
| `POST /api/review/plans/{id}/comment/{cid}` | admin, doctor | `{text}` — edit your own comment (403 for others') |
| `GET /api/review/my-reviews` | doctor | the doctor's review actions from the last 24 h |

State lives in one JSON file (`CLINICAL_AGENT_REVIEW_STORE`, default
`data/treatment_plans.json`).

### Reducing hallucinations

- With a key set, the system prompt requires the model to ground every clinical
  statement in a retrieved source and to say so plainly when the tools return
  nothing, rather than filling the gap from memory.
- With no key, the app performs no synthesis at all — it retrieves and organises
  primary sources and leaves interpretation to the clinician.
- No API keys are needed for the data sources; all are public government APIs
  (openFDA, NCBI E-utilities, RxNav).

### Layout

```
api/
  main.py        FastAPI app; JSON API under /api, static console at /
  agent.py       orchestration — LLM tool-use loop or deterministic retrieval plan
  crew.py        thought generator — Planner/Researcher/Synthesist/Reviewer agents
  lc_agent.py    optional LangChain / LangGraph ReAct evidence-collection engine
  rag_graph.py   optional LangGraph corrective-RAG StateGraph (retrieve→grade→rewrite?→generate)
  tot.py         Tree-of-Thoughts — propose approaches → draft → score → deepen top → synthesise
  review_store.py  file-backed treatment-plan approval workflow (pending → approved → acknowledged)
  auth.py        HMAC-cookie session auth, roles: admin | doctor
  net.py         shared pooled HTTP client + TTL response cache
  fhir.py        sandbox FHIR R4 client — synthetic patient problem list / meds / results as case context
  retrieval.py   one function per live source (openFDA, PubMed, ClinicalTrials.gov, RxNorm, FAERS, MedlinePlus, MedlinePlus Connect, MyHealthfinder, NHS)
  rag.py         local knowledge base — chunk + embed (Voyage/TF-IDF) + search over knowledge/
  kb_load.py     CLI to add a document (pdf/md/txt/html) with provenance and reindex
  models.py      request/response schemas
  config.py      environment configuration
knowledge/       RAG corpus: institutional .md / .txt documents + generated .kb_index.json
webui/           the clinical console (vanilla HTML/CSS/JS, no build step)
```

## MCP server (`mcp_server.py`)

The retrieval layer is also exposed over the **Model Context Protocol**, so any
MCP client (Claude Code, Claude Desktop, …) can call the same evidence sources
directly. stdio transport, no API keys.

| Tool | Source |
| --- | --- |
| `search_local_guidelines(query, k=4)` | Local knowledge base (RAG — TF-IDF over `knowledge/`) |
| `reindex_knowledge_base()` | Rebuilds the local index from `knowledge/` |
| `search_medical_literature(query, max_results=6)` | PubMed (NCBI E-utilities) |
| `search_clinical_trials(query, max_results=6)` | ClinicalTrials.gov API v2 |
| `search_drug_label(drug_name)` | openFDA drug label API |
| `surveillance_signal(term)` | openFDA FAERS — adverse-event trend time series |
| `normalize_drug_name(drug_name)` | RxNorm (RxNav) |
| `search_patient_education(term, max_results=4)` | MedlinePlus (NLM) health-topic pages |
| `search_prevention_guidance(term, max_results=4)` | MyHealthfinder (U.S. health.gov) prevention/screening guidance |
| `search_nhs_conditions(term)` | NHS website Content API — UK conditions A–Z (needs `NHS_API_KEY`) |
| `medlineplus_connect_drug(term, max_results=4)` | MedlinePlus Connect — patient-education entries by drug RxNorm code |
| `gather_clinical_evidence(query, mode)` | runs the workflow-appropriate set of the above in one call |
| `run_clinical_crew(query, mode)` | runs the thought generator (needs `ANTHROPIC_API_KEY` in the server env) and returns the chain of thoughts + cited answer |

Every result carries `human_review_required: true`.

**On WebMD:** WebMD publishes no public API, and scraping it would breach its
terms. In its place the app uses four openly-accessible patient-facing sources —
**MedlinePlus** health topics, **MedlinePlus Connect** (looked up by drug code),
**MyHealthfinder** (health.gov), and the **NHS website Content API** (UK, needs a
free `NHS_API_KEY` — skipped with a note in the trace when absent).

### Register with Claude Code

The repo ships a project-scoped [`.mcp.json`](.mcp.json) — open Claude Code in
this directory and approve it. Or add it explicitly:

```powershell
claude mcp add clinical-agent -- `
  "c:/Users/gradb/ClinicalAgent/.venv/Scripts/python.exe" `
  "c:/Users/gradb/ClinicalAgent/mcp_server.py"
```

### Register with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "clinical-agent": {
      "command": "c:\\Users\\gradb\\ClinicalAgent\\.venv\\Scripts\\python.exe",
      "args": ["c:\\Users\\gradb\\ClinicalAgent\\mcp_server.py"]
    }
  }
}
```

Requires `pip install -r requirements.txt` in `.venv` first (`mcp>=1.2,<2`).

## Node reference server (`server.js`)

An earlier, smaller implementation of the same idea (Claude tool-use + FDA /
PubMed / RxNorm) with a plain chat UI in `public/`.

```powershell
$env:ANTHROPIC_API_KEY = 'your-api-key'
npm install
npm start   # http://127.0.0.1:3000
```

## PowerShell client (`claude.ps1`)

A dependency-free client for Claude's Messages API.

```powershell
$env:ANTHROPIC_API_KEY = 'your-api-key'
.\claude.ps1 -Prompt 'Summarize the clinical handoff workflow in five bullets.'
```

Set `CLAUDE_MODEL` to choose a default model for the session. The key is read
only from the environment and is never written to this project.
