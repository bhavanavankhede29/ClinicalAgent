"use strict";

// One merged workflow. Every query runs the union of all sources and the answer
// is organised into the four sub-workflow sections.
const MERGED_MODE = "comprehensive";

const MODE_LABEL = {
  comprehensive: "Comprehensive review",
  decision_support: "Clinical decision support",
  treatment_research: "Treatment research",
  surveillance: "Disease surveillance",
  narratives: "Clinical narrative",
  treatment_plan: "Personalized treatment plan",
};

const EXAMPLES = [];

const els = {
  transcript: document.getElementById("transcript"),
  emptyState: document.getElementById("empty-state"),
  examples: document.getElementById("examples"),
  composer: document.getElementById("composer"),
  query: document.getElementById("query"),
  send: document.getElementById("send"),
  composerMode: document.getElementById("composer-mode"),
  enginePill: document.getElementById("engine-pill"),
  engineLabel: document.getElementById("engine-label"),
  reviewPill: document.getElementById("review-pill"),
  reviewLabel: document.getElementById("review-label"),
  tabs: document.querySelectorAll(".rail__tab"),
  tabEvidence: document.getElementById("tab-evidence"),
  tabTrace: document.getElementById("tab-trace"),
  railToggle: document.getElementById("rail-toggle"),
  railScrim: document.getElementById("rail-scrim"),
  themeToggle: document.getElementById("theme-toggle"),
  signout: document.getElementById("signout"),
  fhirForm: document.getElementById("fhir-search"),
  fhirName: document.getElementById("fhir-name"),
  fhirResults: document.getElementById("fhir-results"),
  fhirSelected: document.getElementById("fhir-selected"),
  planGenerate: document.getElementById("plan-generate"),
  approvalTracker: document.getElementById("approval-tracker"),
  trackerRefresh: document.getElementById("tracker-refresh"),
  ptModeFhir: document.getElementById("pt-mode-fhir"),
  ptModeManual: document.getElementById("pt-mode-manual"),
  ptFhir: document.getElementById("pt-fhir"),
  ptManual: document.getElementById("pt-manual"),
  manualForm: document.getElementById("manual-form"),
  mName: document.getElementById("m-name"),
  mAge: document.getElementById("m-age"),
  mSex: document.getElementById("m-sex"),
  mSymptoms: document.getElementById("m-symptoms"),
  mMeds: document.getElementById("m-meds"),
  mAllergies: document.getElementById("m-allergies"),
  mHistory: document.getElementById("m-history"),
  mClear: document.getElementById("m-clear"),
  manualApplied: document.getElementById("manual-applied"),
  deepReasoning: document.getElementById("deep-reasoning"),
};

/* ---------- deep reasoning: route through the Tree-of-Thoughts engine ---------- */
(function initDeep() {
  let on = false;
  try { on = localStorage.getItem("ca-deep") === "1"; } catch (_) {}
  if (els.deepReasoning) els.deepReasoning.checked = on;
})();
els.deepReasoning?.addEventListener("change", () => {
  try { localStorage.setItem("ca-deep", els.deepReasoning.checked ? "1" : "0"); } catch (_) {}
});
function queryEndpoint() {
  return els.deepReasoning?.checked ? "/api/query/tot" : "/api/query";
}

// Approval-workflow status → display label + CSS modifier.
const APPROVAL_STATUS = {
  pending_review: ["Pending review", "pending"],
  changes_requested: ["Changes requested", "changes"],
  rejected: ["Rejected by doctor", "changes"],
  approved: ["Approved by doctor", "approved"],
  acknowledged: ["Finalised", "done"],
};

const currentMode = MERGED_MODE;
let selectedPatientId = null;
let selectedPatientName = "";
let patientHasData = true;   // false when a selected patient has no recorded conditions/meds/etc.
let llmConfigured = false;
let latestGate = null;

// Patient context source: "fhir" (sandbox record) or "manual" (typed by hand,
// no FHIR record attached — so nothing from a prior patient can leak in).
let patientMode = "fhir";
let manualContext = "";      // built from the manual form when applied
let manualSummary = "";      // short line for the applied-manual chip
let manualLabel = null;      // the typed name / label, if any

/* ---------- theme ---------- */
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("ca-theme"); } catch (_) {}
  // Default to light; only go dark if the user explicitly chose it before.
  document.documentElement.setAttribute("data-theme", saved === "dark" ? "dark" : "light");
})();
els.themeToggle?.addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.getAttribute("data-theme") === "dark" ||
    (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = isDark ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try { localStorage.setItem("ca-theme", next); } catch (_) {}
});

/* ---------- helpers ---------- */
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// A line/bullet is "critical" if it leads with the ⚠ marker the model is asked
// to use, or opens with a hard safety cue phrase.
const CRIT_CUE = /^(?:[⚠❗🚨]️?\s*)|^\s*\**\s*(?:contraindicat|boxed warning|black[- ]box|do not (?:use|give|co-?administer|combine)|avoid\b|never (?:use|give|combine)|red[- ]flag|life[- ]threatening|fatal\b|serious (?:risk|harm)|danger\b)/i;

function renderMarkdown(md) {
  const inline = (s) =>
    escapeHtml(s)
      // bold spans whose text is a hard safety cue render red, not just bold
      .replace(/\*\*([^*]+)\*\*/g, (_m, t) =>
        CRIT_CUE.test(t) ? `<strong class="crit">${t}</strong>` : `<strong>${t}</strong>`)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      // ⚠-flagged text renders red through to the end of that sentence
      .replace(/([⚠❗🚨]️?\s*[^.;\n]*[.;]?)/gu, '<span class="crit">$1</span>')
      .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_m, g) =>
        g.split(",").map((n) => n.trim())
          .map((n) => `<button class="cite-chip" data-cite="${n}" title="Show source ${n}">${n}</button>`)
          .join(""));

  const lines = String(md || "").replace(/\r/g, "").split("\n");
  let html = "";
  let list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const critClass = (text) => CRIT_CUE.test(text) ? ' class="crit"' : "";

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }
    let m;
    if ((m = line.match(/^(#{2,4})\s+(.*)$/))) {
      closeList();
      const lvl = m[1].length;
      html += `<h${lvl}>${inline(m[2])}</h${lvl}>`;
    } else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); list = "ul"; html += "<ul>"; }
      html += `<li${critClass(m[1])}>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {
      if (list !== "ol") { closeList(); list = "ol"; html += "<ol>"; }
      html += `<li${critClass(m[1])}>${inline(m[1])}</li>`;
    } else {
      closeList();
      html += `<p${critClass(line)}>${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

/* ---------- example prompts ---------- */
function renderExamples() {
  els.examples.innerHTML = "";
  for (const ex of EXAMPLES) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = ex;
    b.addEventListener("click", () => { els.query.value = ex; els.query.focus(); });
    li.appendChild(b);
    els.examples.appendChild(li);
  }
}

/* ---------- patient context (FHIR sandbox) ---------- */
function fhirHint(text) {
  if (els.fhirResults) els.fhirResults.innerHTML = `<li class="fhir-hint">${escapeHtml(text)}</li>`;
}

function renderPatients(patients) {
  els.fhirResults.innerHTML = "";
  const head = document.createElement("li");
  head.className = "fhir-hint";
  head.textContent = `${patients.length} patient${patients.length === 1 ? "" : "s"}`;
  els.fhirResults.appendChild(head);
  patients.forEach((p) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    const meta = [p.gender, p.birth_date].filter(Boolean).join(" · ");
    const chart = p.chart_summary
      ? `<span class="fhir-chart">${escapeHtml(p.chart_summary)}</span>` : "";
    b.innerHTML = `${escapeHtml(p.name || p.id)}<span class="fhir-meta">${escapeHtml(meta || p.id)}</span>${chart}`;
    b.addEventListener("click", () => selectPatient(p));
    li.appendChild(b);
    els.fhirResults.appendChild(li);
  });
}

async function loadPatients(name) {
  fhirHint(name ? "Searching…" : "Loading patients with a recorded chart…");
  try {
    const q = name ? `?name=${encodeURIComponent(name)}` : "?count=40";
    const res = await fetch(`/api/fhir/patients${q}`);
    const data = await res.json();
    const patients = res.ok ? (data.patients || []) : [];
    if (!patients.length) {
      fhirHint(res.ok ? "No patients with a recorded chart matched." : "FHIR server unavailable.");
      return;
    }
    renderPatients(patients);
  } catch (_) {
    fhirHint("FHIR request failed.");
  }
}

if (els.fhirForm) els.fhirForm.addEventListener("submit", (e) => {
  e.preventDefault();
  loadPatients(els.fhirName.value.trim());
});

/* ---------- approval tracker (plans this admin has sent for review) ---------- */
async function loadApprovalTracker() {
  if (!els.approvalTracker) return;
  try {
    const res = await fetch("/api/review/plans?scope=all");
    if (!res.ok) return;
    const all = (await res.json()).plans || [];
    const plans = all.slice(0, 20);   // latest 20 (newest first)
    els.approvalTracker.innerHTML = "";
    if (!all.length) {
      els.approvalTracker.innerHTML =
        `<li class="approval-tracker__empty">Nothing sent yet. Generate a plan and “Send to doctor for approval”.</li>`;
      return;
    }
    for (const p of plans) {
      const [label, mod] = APPROVAL_STATUS[p.status] || [p.status, "pending"];
      const when = (p.reviewed_at || p.created_at || "").slice(0, 16).replace("T", " ");
      const li = document.createElement("li");
      li.className = "approval-tracker__item";
      li.innerHTML = `
        <span class="approval-tracker__pt">${escapeHtml(p.patient_label || "(no patient)")}</span>
        <span class="approval-tracker__row">
          <span class="approval-badge approval-badge--${mod}">${escapeHtml(label)}</span>
          <span class="approval-tracker__when">${escapeHtml(when)}</span>
        </span>
        ${p.reviewed_by ? `<span class="approval-tracker__by">reviewed by ${escapeHtml(p.reviewed_by)}</span>` : ""}`;
      els.approvalTracker.appendChild(li);
    }
    if (all.length > plans.length) {
      const more = document.createElement("li");
      more.className = "approval-tracker__empty";
      more.innerHTML = `+${all.length - plans.length} older · <a class="rail__note--link" href="/approvals">see all</a>`;
      els.approvalTracker.appendChild(more);
    }
  } catch (_) { /* leave whatever is there */ }
}

els.trackerRefresh?.addEventListener("click", loadApprovalTracker);
setInterval(loadApprovalTracker, 30000);

function fhirBlock(label, items) {
  if (!items || !items.length) return "";
  const lis = items.map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  return `<div class="fhir-block"><span class="fhir-block__label">${label}</span><ul>${lis}</ul></div>`;
}

function renderPatientDetail(ctx) {
  const fmtCond = (p) => p.name + (p.status && p.status !== "active" ? ` (${p.status})` : "");
  // fall back to the legacy single list if the split fields aren't present
  const conds = ctx.conditions || (ctx.problems || []).filter((p) => p.kind !== "symptom");
  const syms = ctx.symptoms || (ctx.problems || []).filter((p) => p.kind === "symptom");

  const meds = (ctx.medications || []).map((m) =>
    m.name + (m.dosage ? ` — ${m.dosage}` : ""));
  const results = (ctx.results || []).map((r) =>
    (r.name + " " + (r.value || "")).trim() + (r.date ? ` · ${String(r.date).slice(0, 10)}` : ""));
  const allergies = (ctx.allergies || []).map((a) =>
    a.substance + (a.criticality ? ` (${a.criticality})` : "") + (a.reaction ? ` — ${a.reaction}` : ""));

  const blocks = fhirBlock("Conditions", conds.map(fmtCond))
    + fhirBlock("Symptoms", syms.map(fmtCond))
    + fhirBlock("Medications", meds)
    + fhirBlock("Recent results", results)
    + fhirBlock("Allergies", allergies);

  const head = blocks
    ? `<div class="fhir-panel">${blocks}</div>`
    : `<span class="fhir-selected__summary">No conditions, symptoms, medications, results, or allergies are recorded for this patient in the sandbox. Add the presenting picture below.</span>`;

  const manual = `
    <label class="fhir-manual">
      <span class="fhir-block__label">Presenting symptoms / notes</span>
      <textarea id="fhir-manual" rows="3"
        placeholder="e.g. 3 days of productive cough, fever 38.6, pleuritic chest pain. No PHI."></textarea>
    </label>`;
  return head + manual;
}

async function selectPatient(p) {
  els.fhirResults.innerHTML = "";
  els.fhirSelected.hidden = false;
  els.fhirSelected.innerHTML = `<span class="fhir-selected__name">${escapeHtml(p.name || p.id)}</span>
    <span class="fhir-selected__summary">Loading context…</span>`;
  // Register the selection immediately so a fast "Generate" click while the
  // context request is still in flight still sends the patient id (the backend
  // re-fetches the chart anyway).
  selectedPatientId = p.id;
  selectedPatientName = p.name || p.id;
  patientHasData = true;
  patientMode = "fhir";
  updatePlanAvailability();
  try {
    const res = await fetch(`/api/fhir/patient/${encodeURIComponent(p.id)}/context`);
    const ctx = await res.json();
    if (!res.ok) throw new Error();
    selectedPatientId = p.id;
    selectedPatientName = ctx.patient?.name || p.name || p.id;
    patientHasData = !!(
      (ctx.conditions || []).length || (ctx.symptoms || []).length ||
      (ctx.medications || []).length || (ctx.results || []).length ||
      (ctx.allergies || []).length || (ctx.problems || []).length
    );
    // Also stamp the panel so the id can't be lost to a stale variable.
    els.fhirSelected.dataset.pid = p.id;
    els.fhirSelected.dataset.pname = selectedPatientName;
    updateComposerMode();
    updatePlanAvailability();
    const meta = [ctx.patient?.gender, ctx.patient?.birth_date].filter(Boolean).join(" · ");
    els.fhirSelected.innerHTML = `
      <span class="fhir-selected__name">${escapeHtml(ctx.patient?.name || p.name || p.id)}${
        meta ? ` <span class="fhir-selected__meta">${escapeHtml(meta)}</span>` : ""}</span>
      ${renderPatientDetail(ctx)}
      <button type="button" class="fhir-selected__clear">Clear patient</button>`;
    els.fhirSelected.querySelector(".fhir-selected__clear").addEventListener("click", clearPatient);
  } catch (_) {
    selectedPatientId = null;
    selectedPatientName = "";
    patientHasData = true;
    updateComposerMode();
    updatePlanAvailability();
    els.fhirSelected.innerHTML = `<span class="fhir-selected__summary">Could not load this patient's context.</span>
      <button type="button" class="fhir-selected__clear">Dismiss</button>`;
    els.fhirSelected.querySelector(".fhir-selected__clear").addEventListener("click", clearPatient);
  }
}

function clearPatient() {
  selectedPatientId = null;
  selectedPatientName = "";
  patientHasData = true;
  els.fhirSelected.hidden = true;
  els.fhirSelected.innerHTML = "";
  delete els.fhirSelected.dataset.pid;
  delete els.fhirSelected.dataset.pname;
  updateComposerMode();
  updatePlanAvailability();
  if (patientMode === "fhir") loadPatients(); // restore the default patient list
}

/* ---------- patient mode: FHIR sandbox vs manual entry ---------- */
function setPatientMode(mode) {
  patientMode = mode;
  els.ptModeFhir.classList.toggle("is-active", mode === "fhir");
  els.ptModeManual.classList.toggle("is-active", mode === "manual");
  els.ptFhir.hidden = mode !== "fhir";
  els.ptManual.hidden = mode !== "manual";
  if (mode === "manual") {
    clearPatient();                 // drop any FHIR selection so it can't leak in
  } else {
    manualContext = ""; manualSummary = ""; manualLabel = null;
    els.manualApplied.hidden = true; els.manualApplied.innerHTML = "";
    if (els.fhirResults && !els.fhirResults.children.length) loadPatients();
  }
  updateComposerMode();
  updatePlanAvailability();
}
els.ptModeFhir?.addEventListener("click", () => setPatientMode("fhir"));
els.ptModeManual?.addEventListener("click", () => setPatientMode("manual"));

function buildManualContext() {
  const name = els.mName.value.trim();
  const age = els.mAge.value.trim();
  const sex = els.mSex.value.trim();
  const sym = els.mSymptoms.value.trim();
  const meds = els.mMeds.value.trim();
  const alg = els.mAllergies.value.trim();
  const hx = els.mHistory.value.trim();
  const demo = [name, age && `age ${age}`, sex].filter(Boolean).join(", ");
  const parts = [];
  if (demo) parts.push(`Patient: ${demo}.`);
  if (sym) parts.push(`Presenting complaint: ${sym}`);
  if (meds) parts.push(`Current medications: ${meds}.`);
  if (alg) parts.push(`Allergies: ${alg}.`);
  if (hx) parts.push(`Relevant history / results: ${hx}.`);
  parts.push("(Manually entered — not from any patient record.)");
  const anything = name || age || sex || sym || meds || alg || hx;
  return {
    text: anything ? parts.join(" ") : "",
    label: name || null,
    summary: (name ? name + " · " : "") + [demo && !name ? demo : "", sym].filter(Boolean).join(" · ").slice(0, 120),
  };
}

els.manualForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  const { text, summary, label } = buildManualContext();
  if (!text) { els.mSymptoms.focus(); return; }
  manualContext = text;
  manualSummary = summary || "manual entry";
  manualLabel = label;
  els.manualApplied.hidden = false;
  els.manualApplied.innerHTML = `
    <span class="fhir-selected__name">${escapeHtml(manualLabel || "Manual patient")}</span>
    <span class="fhir-selected__summary">${escapeHtml(manualSummary)}</span>
    <button type="button" class="fhir-selected__clear">Clear</button>`;
  els.manualApplied.querySelector(".fhir-selected__clear").addEventListener("click", clearManual);
  updateComposerMode();
  updatePlanAvailability();
});

function clearManual() {
  manualContext = ""; manualSummary = ""; manualLabel = null;
  els.manualForm.reset();
  els.manualApplied.hidden = true;
  els.manualApplied.innerHTML = "";
  updateComposerMode();
  updatePlanAvailability();
}
els.mClear?.addEventListener("click", clearManual);

/* ---------- what the active patient context contributes to a request ----------
   In manual mode the form is read LIVE — "Apply patient" is just a preview, so a
   query/plan still picks up whatever is typed even if Apply was never clicked. */
function activeContext() {          // -> string for case_context, or ""
  if (patientMode === "manual") return manualContext || buildManualContext().text || "";
  return document.getElementById("fhir-manual")?.value.trim() || "";
}
function activePatientId() {        // -> fhir_patient_id, or null
  if (patientMode !== "fhir") return null;
  // fall back to the panel stamp if the variable was somehow lost
  return selectedPatientId || (els.fhirSelected && !els.fhirSelected.hidden
    ? els.fhirSelected.dataset.pid || null : null);
}
function activePatientLabel() {
  if (patientMode === "manual") {
    const b = buildManualContext();
    return (manualContext || b.text) ? (manualLabel || b.label || b.summary || "Manual entry") : null;
  }
  return selectedPatientName
    || (els.fhirSelected && !els.fhirSelected.hidden ? els.fhirSelected.dataset.pname : null)
    || null;
}
function hasPatientContext() {
  return !!(activePatientId()
    || (patientMode === "manual" && (manualContext || buildManualContext().text)));
}

// Query mode: a FHIR patient with no recorded data -> a single focused answer,
// not the four-section comprehensive review. Manual entry always carries the
// clinician's picture, so it keeps the normal mode.
function effectiveMode() {
  return (patientMode === "fhir" && selectedPatientId && !patientHasData)
    ? "decision_support" : currentMode;
}
function updateComposerMode() {
  if (!els.composerMode) return;
  els.composerMode.textContent =
    (patientMode === "fhir" && selectedPatientId && !patientHasData)
      ? "Focused — no patient record" : MODE_LABEL[currentMode];
}

// Both actions stay enabled regardless — the clinician decides what to retrieve.
function updatePlanAvailability() {
  if (!els.planGenerate) return;
  els.planGenerate.title = hasPatientContext()
    ? `Patient context: ${activePatientLabel()}`
    : "No patient context — you'll be asked to add one.";
}

/* ---------- evidence rail ---------- */
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

function activateTab(name) {
  els.tabs.forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", String(on));
  });
  els.tabEvidence.classList.toggle("is-hidden", name !== "evidence");
  els.tabTrace.classList.toggle("is-hidden", name !== "trace");
}

function sparkline(series) {
  const NS = "http://www.w3.org/2000/svg";
  const w = 320, h = 46, pad = 2;
  const max = Math.max(1, ...series.map((d) => d.count));
  const bw = (w - pad * 2) / series.length;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("class", "spark");
  svg.setAttribute("preserveAspectRatio", "none");
  let inner = `<line x1="0" y1="${h - 9}" x2="${w}" y2="${h - 9}"/>`;
  series.forEach((d, i) => {
    const bh = Math.max(1, (d.count / max) * (h - 14));
    const x = pad + i * bw;
    inner += `<rect x="${x + 1}" y="${h - 9 - bh}" width="${Math.max(1, bw - 2)}" height="${bh}"><title>${escapeHtml(d.period)}: ${d.count}</title></rect>`;
    if (i === 0 || i === series.length - 1) {
      inner += `<text x="${i === 0 ? 1 : w - 1}" y="${h - 1}" text-anchor="${i === 0 ? "start" : "end"}">${escapeHtml(d.period)}</text>`;
    }
  });
  svg.innerHTML = inner;
  return svg;
}

function evidenceCard(c) {
  const el = document.createElement("div");
  el.className = "ev-card";
  el.id = `ev-${c.id}`;
  const kindLabel = c.source_kind.replace(/-/g, " ");
  const scoreChip = (typeof c.score === "number")
    ? `<span class="ev-score" title="${escapeHtml(c.retrieval || "similarity")} similarity">${c.score.toFixed(3)}</span>`
    : "";
  let linkText = "Open source";
  if (c.url) {
    if (c.url.startsWith("/api/kb/")) linkText = "Open local document";
    else { try { linkText = new URL(c.url).hostname.replace(/^www\./, ""); } catch (_) {} }
  }
  const titleHtml = c.url
    ? `<a class="ev-card__title ev-card__title--link" href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(c.title)}</a>`
    : `<div class="ev-card__title">${escapeHtml(c.title)}</div>`;
  el.innerHTML = `
    <div class="ev-card__top">
      <span class="ev-id">[${c.id}]</span>
      <span class="ev-kind k-${c.source_kind}">${kindLabel}</span>
      ${scoreChip}
    </div>
    ${titleHtml}
    <div class="ev-card__meta">${escapeHtml(c.source)}${c.published ? " · " + escapeHtml(c.published) : ""} · retrieved ${escapeHtml(c.retrieved_at)}</div>
    <div class="ev-card__snippet">${escapeHtml(c.snippet)}</div>
    ${c.url
      ? `<a class="ev-card__link" href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(linkText)} ↗</a>`
      : `<span class="ev-card__link ev-card__link--local">no link</span>`}
  `;
  if (Array.isArray(c.series) && c.series.length) el.appendChild(sparkline(c.series));
  return el;
}

// External sources first, local knowledge-base hits last (ids are unchanged).
const localLast = (a, b) =>
  (a.source_kind === "local-knowledge") - (b.source_kind === "local-knowledge");

function renderEvidence(citations, query) {
  els.tabEvidence.innerHTML = "";
  const head = document.createElement("p");
  head.className = "rail__note";
  head.textContent = citations.length
    ? `${citations.length} source(s) for: “${query}”`
    : `No sources retrieved for: “${query}”`;
  els.tabEvidence.appendChild(head);
  [...citations].sort(localLast).forEach((c) => els.tabEvidence.appendChild(evidenceCard(c)));
}

function renderTrace(trace) {
  els.tabTrace.innerHTML = "";
  if (!trace.length) {
    els.tabTrace.innerHTML = `<p class="rail__placeholder">No retrieval steps recorded.</p>`;
    return;
  }
  for (const s of trace) {
    const div = document.createElement("div");
    div.className = `trace-step ${s.status === "error" ? "err" : "ok"}`;
    div.innerHTML = `
      <span class="trace-step__head">${s.step}. ${escapeHtml(s.tool)} — ${s.result_count} result(s) · ${s.latency_ms} ms · ${escapeHtml(s.status)}</span>
      <span class="trace-step__params">via ${escapeHtml(s.source_api)} · ${escapeHtml(JSON.stringify(s.params))}</span>
      ${s.note ? `<span class="trace-step__note">${escapeHtml(s.note)}</span>` : ""}
    `;
    els.tabTrace.appendChild(div);
  }
}

function focusCitation(id) {
  activateTab("evidence");
  if (matchMedia("(max-width: 1120px)").matches) openRail(true);
  const card = document.getElementById(`ev-${id}`);
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.remove("flash");
    void card.offsetWidth;
    card.classList.add("flash");
  }
}

/* ---------- transcript ---------- */
function buildRecord(data, reviewed) {
  const L = [];
  L.push(`CLINICAL AGENT — ${MODE_LABEL[data.mode] || data.mode}`);
  L.push(`Generated: ${data.generated_at}`);
  L.push(`Engine: ${data.llm.used
    ? `${data.llm.model} (synthesis, ${data.llm.tool_rounds} retrieval round(s))`
    : `evidence-only retrieval${data.llm.reason ? " — " + data.llm.reason : ""}`}`);
  L.push(`Clinician review: ${reviewed ? "CONFIRMED " + new Date().toISOString() : "NOT CONFIRMED"}`);
  L.push("");
  L.push(`QUERY: ${data.query}`);
  L.push("");
  L.push("RESPONSE:");
  L.push(data.answer_markdown);
  L.push("");
  L.push("SOURCES:");
  data.citations.forEach((c) => {
    L.push(`[${c.id}] ${c.title}`);
    L.push(`    ${c.source} | ${c.published || "n.d."} | retrieved ${c.retrieved_at}`);
    if (c.url) L.push(`    ${c.url}`);
  });
  L.push("");
  L.push("RETRIEVAL TRACE:");
  data.retrieval_trace.forEach((s) => {
    L.push(`  ${s.step}. ${s.tool} via ${s.source_api} — ${s.result_count} result(s), ${s.latency_ms}ms, ${s.status}${s.note ? " — " + s.note : ""}`);
  });
  L.push("");
  L.push("Decision support only. Not a substitute for clinical judgement. Verify against current prescribing information and guidelines before any patient-impacting action.");
  return L.join("\n");
}

function setReviewPill(done) {
  els.reviewPill.className = `pill ${done ? "pill--ok" : "pill--caution"}`;
  els.reviewLabel.textContent = done ? "Reviewed" : "Review pending";
}

function renderAnswerInto(data) {
  const wrap = document.createElement("article");
  wrap.className = "exchange";

  const q = document.createElement("div");
  q.className = "q-line";
  q.innerHTML = `<span class="q-mode">${escapeHtml(MODE_LABEL[data.mode] || data.mode)}</span>${escapeHtml(data.query)}`;
  wrap.appendChild(q);

  const answer = document.createElement("div");
  answer.className = "answer";
  const engineTag = data.llm.used
    ? `<span class="tag tag--synthesis">synthesis</span>`
    : `<span class="tag tag--evidence-only">evidence-only</span>`;
  answer.innerHTML = `
    <div class="answer__head">
      ${engineTag}
      <span class="tag">${data.citations.length} sources</span>
      <span class="tag">${data.retrieval_trace.length} retrieval steps</span>
      ${data.llm.used ? `<span class="tag">${data.llm.tool_rounds} rounds</span>` : ""}
      <span style="margin-left:auto">${escapeHtml(data.generated_at)}</span>
    </div>
    <div class="prose">${renderMarkdown(data.answer_markdown)}</div>
    <div class="answer__foot"></div>
  `;

  const foot = answer.querySelector(".answer__foot");

  if (data.citations.length) {
    const row = document.createElement("div");
    row.className = "cite-row";
    row.innerHTML = `<span class="cite-row__label">Cited sources</span>`;
    [...data.citations].sort(localLast).forEach((c) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "cite-pill";
      pill.innerHTML = `<b>[${c.id}]</b> ${escapeHtml(c.title.slice(0, 60))}${c.title.length > 60 ? "…" : ""}`;
      pill.addEventListener("click", () => focusCitation(c.id));
      row.appendChild(pill);
    });
    foot.appendChild(row);
  }

  const gate = document.createElement("label");
  gate.className = "review-gate";
  gate.innerHTML = `
    <input type="checkbox" />
    <span class="review-gate__text">I have reviewed the evidence and this output against current guidance. Required before use in any patient-impacting decision.</span>
  `;
  const checkbox = gate.querySelector("input");
  foot.appendChild(gate);

  const actions = document.createElement("div");
  actions.className = "review-actions";
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "ghost-btn";
  copyBtn.textContent = "Copy for record";
  copyBtn.disabled = true;
  copyBtn.addEventListener("click", async () => {
    const text = buildRecord(data, checkbox.checked);
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy for record"), 1500);
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy for record"), 1500);
    }
  });
  actions.appendChild(copyBtn);

  // A real treatment plan can be sent to a doctor for approval.
  if (data.mode === "treatment_plan" && data.llm && data.llm.used) {
    const sendBtn = document.createElement("button");
    sendBtn.type = "button";
    sendBtn.className = "send-btn";
    sendBtn.textContent = "Send to doctor for approval";
    sendBtn.addEventListener("click", async () => {
      sendBtn.disabled = true;
      sendBtn.textContent = "Sending…";
      try {
        const res = await fetch("/api/review/plans", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            query: data.query,
            answer_markdown: data.answer_markdown,
            citations: (data.citations || []).map((c) => ({
              id: c.id, title: c.title, source: c.source, url: c.url,
            })),
            fhir_patient_id: activePatientId(),
            patient_label: activePatientLabel(),
          }),
        });
        if (res.status === 401) { window.location.href = "/"; return; }
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          sendBtn.textContent = d.detail || "Send failed — retry";
          sendBtn.disabled = false;
          return;
        }
        sendBtn.textContent = "Sent to doctor ✓";
        sendBtn.classList.remove("send-btn");
        sendBtn.classList.add("ghost-btn");
        loadApprovalTracker();
      } catch (_) {
        sendBtn.textContent = "Network error — retry";
        sendBtn.disabled = false;
      }
    });
    actions.appendChild(sendBtn);
  }

  foot.appendChild(actions);

  checkbox.addEventListener("change", () => {
    gate.classList.toggle("is-done", checkbox.checked);
    copyBtn.disabled = !checkbox.checked;
    if (checkbox === latestGate) setReviewPill(checkbox.checked);
  });
  latestGate = checkbox;
  setReviewPill(false);

  wrap.appendChild(answer);

  answer.querySelectorAll(".cite-chip").forEach((chip) => {
    chip.addEventListener("click", () => focusCitation(chip.dataset.cite));
  });

  return wrap;
}

function addExchange(data) {
  if (els.emptyState) { els.emptyState.remove(); els.emptyState = null; }
  const wrap = renderAnswerInto(data);
  els.transcript.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "start" });
  renderEvidence(data.citations, data.query);
  renderTrace(data.retrieval_trace);
}

function addError(message) {
  if (els.emptyState) { els.emptyState.remove(); els.emptyState = null; }
  const wrap = document.createElement("article");
  wrap.className = "exchange";
  wrap.innerHTML = `
    <div class="answer">
      <div class="answer__head"><span class="tag tag--evidence-only">request failed</span></div>
      <div class="prose"><p>${escapeHtml(message)}</p></div>
    </div>`;
  els.transcript.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- submit ---------- */
els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const typed = els.query.value.trim();
  // An empty box with a patient in context still means something — run a
  // comprehensive review of that patient rather than silently doing nothing.
  const query = typed.length >= 3 ? typed
    : hasPatientContext() ? `Comprehensive review for ${activePatientLabel() || "this patient"}.`
    : "";
  if (query.length < 3) {
    els.query.focus();
    els.query.classList.remove("input-flash");
    void els.query.offsetWidth;
    els.query.classList.add("input-flash");
    return;
  }

  const payload = {
    mode: effectiveMode(),
    query,
  };
  const pid = activePatientId();
  if (pid) payload.fhir_patient_id = pid;
  const ctx = activeContext();
  if (ctx) payload.case_context = ctx;

  const deep = els.deepReasoning?.checked;
  els.send.disabled = true;
  els.send.textContent = deep ? "Deep reasoning…" : "Retrieving…";
  els.query.value = "";

  try {
    const res = await fetch(queryEndpoint(), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.status === 401) { window.location.href = "/"; return; }
    const data = await res.json();
    if (!res.ok) {
      addError(data.detail || `Request failed (${res.status}).`);
    } else {
      addExchange(data);
    }
  } catch (err) {
    addError(err.message || "Network error.");
  } finally {
    els.send.disabled = false;
    els.send.textContent = llmConfigured ? "Detailed References" : "Retrieve evidence";
    els.query.focus();
  }
});

els.planGenerate?.addEventListener("click", async () => {
  // Whatever the clinician typed in the query box is real input — use it
  // instead of silently discarding it in favour of a generic default.
  const typed = els.query.value.trim();
  const pid = activePatientId();
  const ctx = activeContext();
  if (!pid && !ctx && !typed) {
    addError("No patient is attached to this request. Pick a patient in the panel "
      + "(click them again if one is shown), switch to Manual entry and fill in the "
      + "details, or type the presenting picture in the box above.");
    return;
  }
  const payload = {
    mode: "treatment_plan",
    query: typed || `Personalized treatment plan for ${activePatientLabel() || "this patient"}.`,
  };
  if (pid) payload.fhir_patient_id = pid;
  if (ctx) payload.case_context = ctx;

  els.planGenerate.disabled = true;
  els.planGenerate.textContent = els.deepReasoning?.checked ? "Deep reasoning…" : "Generating…";
  els.query.value = "";

  try {
    const res = await fetch(queryEndpoint(), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.status === 401) { window.location.href = "/"; return; }
    const data = await res.json();
    if (!res.ok) {
      addError(data.detail || `Request failed (${res.status}).`);
    } else {
      addExchange(data);
    }
  } catch (err) {
    addError(err.message || "Network error.");
  } finally {
    els.planGenerate.disabled = false;
    els.planGenerate.textContent = "Generate treatment plan";
  }
});

els.query.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.planGenerate?.click(); // Enter in the query box defaults to Generate treatment plan
  }
});

els.signout?.addEventListener("click", async () => {
  try { await fetch("/api/logout", { method: "POST" }); } catch (_) {}
  window.location.href = "/";
});

/* ---------- mobile rail ---------- */
function openRail(open) {
  document.body.classList.toggle("rail-open", open);
  els.railToggle?.setAttribute("aria-expanded", String(open));
  if (els.railScrim) els.railScrim.hidden = !open;
}
els.railToggle?.addEventListener("click", () => openRail(!document.body.classList.contains("rail-open")));
els.railScrim?.addEventListener("click", () => openRail(false));

/* ---------- init ---------- */
async function init() {
  renderExamples();
  updateComposerMode();
  updatePlanAvailability();
  if (els.fhirResults) loadPatients();
  loadApprovalTracker();
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    llmConfigured = !!h.llm_configured;
    if (llmConfigured) {
      els.enginePill.className = "pill pill--accent";
      els.engineLabel.textContent = `Synthesis · ${h.model}`;
      els.send.textContent = "Detailed References";
    } else {
      els.enginePill.className = "pill pill--muted";
      els.engineLabel.textContent = "Evidence-only";
      els.send.textContent = "Retrieve evidence";
    }
  } catch (_) {
    els.enginePill.className = "pill pill--caution";
    els.engineLabel.textContent = "API unreachable";
  }
}

init();
