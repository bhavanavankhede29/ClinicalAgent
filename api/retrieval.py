"""Evidence sources.

Each function queries one public, authoritative medical API directly over HTTPS.
No API keys are required. Every function returns:

    (raw_citations, source_api, note)

where raw_citations is a list of dicts shaped for models.Citation (minus the id,
which the orchestrator assigns), optionally carrying:
    _detail  -- a richer payload handed to the LLM but not shown verbatim in the UI
    _series  -- [(period, count), ...] for time-series evidence
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
FDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
RXNORM_RXCUI = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_PROPS = "https://rxnav.nlm.nih.gov/REST/rxcui/{}/properties.json"
CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
MYHEALTHFINDER_URL = "https://odphp.health.gov/myhealthfinder/api/v4/topicsearch.json"
NHS_CONDITIONS_URL = "https://api.nhs.uk/conditions/{}"
MEDLINEPLUS_CONNECT_URL = "https://connect.medlineplus.gov/service"
RXNORM_OID = "2.16.840.1.113883.6.88"  # code system OID for RxNorm, for MedlinePlus Connect

_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "with", "and", "or", "to", "is", "are",
    "vs", "versus", "risk", "dose", "dosing", "doses", "treatment", "therapy", "management",
    "patient", "patients", "adult", "adults", "use", "using", "child", "children",
    "adverse", "event", "events", "trend", "trends", "report", "reports", "reporting",
    "signal", "safety", "current", "evidence", "recent", "data", "rate", "rates",
}


def primary_term(query: str) -> str:
    """Cheap entity guess for APIs that need a single drug/condition token.

    The full query is still used for literature search; this only feeds the
    label / nomenclature / pharmacovigilance lookups, and the retrieval trace
    records exactly what was searched.
    """
    tokens = [t.strip("-") for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", query)]
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return query.strip()
    caps = [t for t in tokens if t[0].isupper()]
    if len(caps) == 1:
        return caps[0]
    if len(caps) > 1:
        non_first = [t for t in caps if t != tokens[0]]
        return max(non_first or caps, key=len)
    candidates = [t for t in tokens if t.lower() not in _STOP]
    return max(candidates, key=len) if candidates else tokens[0]


def _clean(value, limit: int = 500) -> str:
    if not value:
        return ""
    text = " ".join(value) if isinstance(value, list) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


async def search_literature(client: httpx.AsyncClient, *, query: str, retmax: int = 6):
    """PubMed peer-reviewed literature — treatment research, epidemiology, narrative background."""
    from .config import CONTACT

    common = {"db": "pubmed", "retmode": "json", "tool": "clinical-agent", "email": CONTACT}
    r = await client.get(PUBMED_ESEARCH, params={**common, "retmax": retmax, "sort": "relevance", "term": query})
    if r.status_code != 200:
        return [], "NCBI PubMed E-utilities", f"PubMed search failed ({r.status_code})."
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return [], "NCBI PubMed E-utilities", "No PubMed articles matched this query."

    s = await client.get(PUBMED_ESUMMARY, params={**common, "id": ",".join(ids)})
    result = s.json().get("result", {}) if s.status_code == 200 else {}

    cites = []
    for pmid in ids:
        item = result.get(pmid)
        if not item:
            continue
        authors = ", ".join(a.get("name", "") for a in item.get("authors", [])[:3])
        journal = item.get("fulljournalname") or item.get("source", "")
        pubtypes = [p for p in item.get("pubtype", []) if p and p != "Journal Article"]
        cites.append({
            "source": f"PubMed · {item.get('source', 'journal')} · {item.get('pubdate', 'n.d.')}",
            "source_kind": "literature",
            "title": _clean(item.get("title", "(untitled)"), 240),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "published": item.get("pubdate"),
            "snippet": _clean(f"{authors}. {journal}. "
                              f"{('; '.join(pubtypes) + '. ') if pubtypes else ''}PMID {pmid}. "
                              f"Abstract and full text via the link."),
            "_detail": {"pmid": pmid, "journal": journal, "publication_types": item.get("pubtype", [])},
        })
    return cites, "NCBI PubMed E-utilities", None


async def search_drug_label(client: httpx.AsyncClient, *, query: str, limit: int = 3):
    """FDA structured product labelling — indications, dosing, contraindications, warnings, interactions."""
    async def _attempt(field: str) -> list:
        r = await client.get(FDA_LABEL_URL,
                             params={"search": f'openfda.{field}:"{query}"', "limit": limit})
        return r.json().get("results", []) if r.status_code == 200 else []

    # Try generic- and brand-name matches concurrently; prefer the generic hit.
    attempts = await asyncio.gather(_attempt("generic_name"), _attempt("brand_name"),
                                    return_exceptions=True)
    results = next((a for a in attempts if isinstance(a, list) and a), None)
    if results:
        cites = []
        for item in results:
            fda = item.get("openfda", {})
            name = (fda.get("brand_name") or fda.get("generic_name") or ["Unknown product"])[0]
            set_id = (fda.get("spl_set_id") or [None])[0]
            url = (f"https://labels.fda.gov/getSPLViewFromSPLSetId.cfm?setId={set_id}"
                   if set_id else "https://labels.fda.gov/")
            boxed = _clean(item.get("boxed_warning"), 320)
            snippet = _clean(item.get("indications_and_usage") or item.get("purpose")
                             or item.get("description"), 420)
            if boxed:
                snippet = f"⚠ BOXED WARNING — {boxed}  |  {snippet}"
            cites.append({
                "source": f"FDA label · {name}",
                "source_kind": "fda-label",
                "title": f"{name} — FDA structured product label",
                "url": url,
                "published": (item.get("effective_time") or [None])[0],
                "snippet": snippet,
                "_detail": {
                    "dosage_and_administration": _clean(item.get("dosage_and_administration"), 900),
                    "contraindications": _clean(item.get("contraindications"), 700),
                    "warnings_and_cautions": _clean(item.get("warnings_and_cautions") or item.get("warnings"), 900),
                    "drug_interactions": _clean(item.get("drug_interactions"), 900),
                    "use_in_specific_populations": _clean(item.get("use_in_specific_populations"), 700),
                },
            })
        return cites, "openFDA drug label API", None
    return [], "openFDA drug label API", f'No FDA label matched "{query}" (try the generic name).'


async def normalize_drug(client: httpx.AsyncClient, *, drug_name: str):
    """RxNorm — resolve a drug name to a standardized concept (RxCUI, canonical name, synonyms)."""
    r = await client.get(RXNORM_RXCUI, params={"name": drug_name})
    if r.status_code != 200:
        return [], "RxNorm (U.S. National Library of Medicine)", f"RxNorm lookup failed ({r.status_code})."
    rxcui = (r.json().get("idGroup", {}).get("rxnormId") or [None])[0]
    if not rxcui:
        return [], "RxNorm (U.S. National Library of Medicine)", f'No exact RxNorm concept for "{drug_name}".'

    p = await client.get(RXNORM_PROPS.format(rxcui))
    props = p.json().get("properties", {}) if p.status_code == 200 else {}
    cite = {
        "source": f"RxNorm concept {rxcui}",
        "source_kind": "nomenclature",
        "title": f"{props.get('name', drug_name)} — RxNorm normalized concept",
        "url": f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
        "published": None,
        "snippet": _clean(f"RxCUI {rxcui} · term type {props.get('tty', '?')} · "
                          f"synonym: {props.get('synonym') or '—'}"),
        "_detail": props,
    }
    return [cite], "RxNorm (U.S. National Library of Medicine)", None


async def surveillance_signal(client: httpx.AsyncClient, *, term: str):
    """FDA FAERS — adverse-event report volume over time for a drug name or a reaction term.

    A rising or anomalous report trend is a pharmacovigilance / surveillance signal,
    not proof of causation.
    """
    upper = term.upper()
    variants = [
        f'patient.drug.medicinalproduct:"{upper}"',
        f'patient.drug.openfda.generic_name.exact:"{upper}"',
        f'patient.drug.openfda.brand_name.exact:"{upper}"',
        f'patient.reaction.reactionmeddrapt:"{term}"',
    ]

    async def _fetch(search: str) -> list:
        r = await client.get(FDA_EVENT_URL, params={"search": search, "count": "receivedate"})
        return r.json().get("results", []) if r.status_code == 200 else []

    # Query all field variants concurrently, then take the first (by priority) that hit.
    all_rows = await asyncio.gather(*(_fetch(v) for v in variants), return_exceptions=True)
    for search, rows in zip(variants, all_rows):
        if not isinstance(rows, list) or not rows:
            continue

        by_year: dict[str, int] = {}
        for row in rows:
            year = str(row.get("time", ""))[:4]
            if year:
                by_year[year] = by_year.get(year, 0) + int(row.get("count", 0))
        series = sorted(by_year.items())
        if not series:
            continue
        total = sum(v for _, v in series)
        recent = series[-8:]
        trend = ", ".join(f"{y}: {n:,}" for y, n in recent)
        cite = {
            "source": "FDA FAERS (Adverse Event Reporting System)",
            "source_kind": "pharmacovigilance",
            "title": f"Adverse-event report trend — “{term}”",
            "url": "https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers",
            "published": recent[-1][0],
            "snippet": _clean(f"FAERS report volume by receipt year — {trend} (cumulative {total:,}). "
                              f"Spontaneous reports; no denominator, no causation implied."),
            "_detail": {"matched_on": search, "years": dict(series)},
            "_series": recent,
        }
        return [cite], "openFDA FAERS API", None
    return [], "openFDA FAERS API", f'No FAERS reports found for "{term}".'


async def surveillance_top_reported(client: httpx.AsyncClient, *, limit: int = 15):
    """FDA FAERS — the most frequently reported adverse-event/reaction terms across ALL
    reports (no drug named). Use this when a clinician asks a general "what's being
    reported" / "list of [diseases/reactions] reported" question with no specific drug,
    condition, or reaction named — surveillance_signal needs one of those; this doesn't.

    These are reaction terms coded on drug-exposure reports (MedDRA preferred terms),
    not a population disease registry — read as "what accompanies a drug report", not
    "diseases in the population".
    """
    limit = max(1, min(limit, 30))
    r = await client.get(FDA_EVENT_URL, params={
        "count": "patient.reaction.reactionmeddrapt.exact", "limit": limit,
    })
    if r.status_code != 200:
        return [], "openFDA FAERS API", f"FAERS aggregate query failed ({r.status_code})."
    rows = r.json().get("results", []) or []
    if not rows:
        return [], "openFDA FAERS API", "No FAERS aggregate data returned."
    ranked = ", ".join(f"{row.get('term')} ({row.get('count'):,})" for row in rows)
    cite = {
        "source": "FDA FAERS (Adverse Event Reporting System)",
        "source_kind": "pharmacovigilance",
        "title": f"Most frequently reported adverse-event terms (top {len(rows)}, all drugs)",
        "url": "https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers",
        "published": None,
        "snippet": _clean(f"Ranked by report count across the entire FAERS database — {ranked}. "
                          "Spontaneous reports tied to drug exposure; no denominator, no causation, "
                          "not a disease-surveillance registry."),
        "_detail": {"ranked_terms": [{"term": row.get("term"), "count": row.get("count")} for row in rows]},
    }
    return [cite], "openFDA FAERS API", None


async def search_clinical_trials(client: httpx.AsyncClient, *, query: str, page_size: int = 6):
    """ClinicalTrials.gov (API v2) — registered interventional and observational studies.

    Use for treatment research: what is being or has been trialled for a condition
    or drug, the phase, recruitment status, and sponsor.
    """
    params = {
        "query.term": query,
        "pageSize": max(1, min(page_size, 20)),
        "sort": "LastUpdatePostDate:desc",
        "format": "json",
    }
    r = await client.get(CTGOV_URL, params=params)
    if r.status_code != 200:
        return [], "ClinicalTrials.gov API v2", f"ClinicalTrials.gov search failed ({r.status_code})."
    studies = r.json().get("studies", [])
    if not studies:
        return [], "ClinicalTrials.gov API v2", f'No registered studies matched "{query}".'

    cites = []
    for study in studies:
        ps = study.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        status = ps.get("statusModule", {})
        design = ps.get("designModule", {})
        arms = ps.get("armsInterventionsModule", {})
        nct = ident.get("nctId", "")
        phases = ", ".join(design.get("phases", []) or []) or "N/A"
        conditions = ", ".join(ps.get("conditionsModule", {}).get("conditions", [])[:4])
        interventions = ", ".join(
            i.get("name", "") for i in arms.get("interventions", [])[:4] if i.get("name")
        )
        overall = status.get("overallStatus", "unknown")
        cites.append({
            "source": f"ClinicalTrials.gov · {nct} · {overall}",
            "source_kind": "clinical-trial",
            "title": _clean(ident.get("briefTitle") or ident.get("officialTitle") or nct, 240),
            "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "https://clinicaltrials.gov/",
            "published": (status.get("lastUpdatePostDateStruct", {}) or {}).get("date"),
            "snippet": _clean(
                f"Status: {overall}. Phase: {phases}. "
                f"{('Conditions: ' + conditions + '. ') if conditions else ''}"
                f"{('Interventions: ' + interventions + '. ') if interventions else ''}"
                f"Type: {design.get('studyType', 'n/a')}."
            ),
            "_detail": {
                "nct_id": nct,
                "overall_status": overall,
                "phases": design.get("phases", []),
                "study_type": design.get("studyType"),
                "enrollment": (design.get("enrollmentInfo", {}) or {}).get("count"),
                "lead_sponsor": (ps.get("sponsorCollaboratorsModule", {})
                                 .get("leadSponsor", {}) or {}).get("name"),
                "start_date": (status.get("startDateStruct", {}) or {}).get("date"),
            },
        })
    return cites, "ClinicalTrials.gov API v2", None


_TAG_RE = re.compile(r"<[^>]+>")


async def search_consumer_health(client: httpx.AsyncClient, *, term: str, retmax: int = 4):
    """MedlinePlus (NLM) health-topic pages — plain-language information on conditions,
    symptoms, tests, and treatments.

    This is the authoritative open substitute for consumer-health sites such as
    WebMD, which do not offer a public API. Content is written for patients and
    caregivers, not as clinical guidance.
    """
    params = {"db": "healthTopics", "term": term, "retmax": max(1, min(retmax, 10))}
    r = await client.get(MEDLINEPLUS_URL, params=params)
    if r.status_code != 200:
        return [], "MedlinePlus Web Service (NLM)", f"MedlinePlus search failed ({r.status_code})."

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as exc:
        return [], "MedlinePlus Web Service (NLM)", f"Could not parse MedlinePlus response: {exc}"

    docs = root.findall(".//document")
    if not docs:
        return [], "MedlinePlus Web Service (NLM)", f'No MedlinePlus health topic matched "{term}".'

    cites = []
    for doc in docs:
        fields = {c.get("name"): "".join(c.itertext()) for c in doc.findall("content")}
        title = _TAG_RE.sub("", fields.get("title", "")).strip() or "(untitled topic)"
        summary = _TAG_RE.sub("", fields.get("FullSummary", "") or fields.get("snippet", ""))
        cites.append({
            "source": "MedlinePlus health topic (NLM)",
            "source_kind": "patient-education",
            "title": title,
            "url": doc.get("url") or "https://medlineplus.gov/",
            "published": None,
            "snippet": _clean(summary, 500) or "Plain-language health topic page.",
            "_detail": {"also_called": fields.get("altTitle"), "groups": fields.get("groupName")},
        })
    return cites, "MedlinePlus Web Service (NLM)", None


async def search_myhealthfinder(client: httpx.AsyncClient, *, term: str, limit: int = 4):
    """MyHealthfinder (U.S. health.gov) — prevention and screening guidance written
    for patients: what to do, when, and why. Free, no key."""
    r = await client.get(MYHEALTHFINDER_URL, params={"keyword": term})
    if r.status_code != 200:
        return [], "MyHealthfinder API (health.gov)", f"MyHealthfinder search failed ({r.status_code})."
    resources = (r.json().get("Result", {}).get("Resources", {}) or {}).get("Resource", []) or []
    if not resources:
        return [], "MyHealthfinder API (health.gov)", f'No MyHealthfinder topic matched "{term}".'

    cites = []
    for res in resources[: max(1, min(limit, 8))]:
        sections = ((res.get("Sections", {}) or {}).get("section", []) or [])
        body = " ".join(_TAG_RE.sub(" ", s.get("Content", "") or "") for s in sections[:2])
        cats = res.get("Categories", "")
        cites.append({
            "source": f"MyHealthfinder · health.gov{(' · ' + cats) if cats else ''}",
            "source_kind": "patient-education",
            "title": _clean(res.get("Title", "(untitled)"), 200),
            "url": res.get("AccessibleVersion") or res.get("LastUpdate") or "https://health.gov/myhealthfinder",
            "published": res.get("LastUpdate"),
            "snippet": _clean(body, 500) or "Patient-facing prevention/screening guidance.",
            "_detail": {"categories": cats, "section_titles": [s.get("Title") for s in sections][:6]},
        })
    return cites, "MyHealthfinder API (health.gov)", None


async def search_nhs_conditions(client: httpx.AsyncClient, *, term: str):
    """NHS website Content API — UK conditions A-Z, patient-facing. Requires a free
    subscription key in NHS_API_KEY; skipped with a note if it is not set."""
    from .config import NHS_API_KEY

    source = "NHS website Content API"
    if not NHS_API_KEY:
        return [], source, "NHS API key not configured (set NHS_API_KEY); source skipped."

    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
    if not slug:
        return [], source, f'Could not derive an NHS conditions slug from "{term}".'

    r = await client.get(
        NHS_CONDITIONS_URL.format(slug),
        headers={"subscription-key": NHS_API_KEY, "accept": "application/json"},
    )
    if r.status_code == 404:
        return [], source, f'No NHS conditions page for "{slug}".'
    if r.status_code != 200:
        return [], source, f"NHS Content API returned {r.status_code}."

    data = r.json()
    name = data.get("name") or term.title()
    desc = data.get("description") or ""
    # First readable paragraph from the page body, if present.
    para = ""
    for block in data.get("mainEntityOfPage", []) or []:
        if block.get("@type") == "WebPageElement":
            for sub in block.get("mainEntityOfPage", []) or []:
                txt = _TAG_RE.sub("", sub.get("text", "") or "")
                if len(txt) > 60:
                    para = txt
                    break
        if para:
            break

    cite = {
        "source": "NHS website (nhs.uk)",
        "source_kind": "patient-education",
        "title": _clean(name, 200),
        "url": data.get("url") or data.get("@id") or f"https://www.nhs.uk/conditions/{slug}/",
        "published": data.get("dateModified") or data.get("lastReviewed"),
        "snippet": _clean(f"{desc} {para}".strip(), 500) or "NHS patient-facing conditions page.",
        "_detail": {"genre": data.get("genre"), "about": data.get("about")},
    }
    return [cite], source, None


async def medlineplus_connect_drug(client: httpx.AsyncClient, *, term: str, limit: int = 4):
    """MedlinePlus Connect — patient-education entries for a drug, looked up by its
    RxNorm code. Connect is code-driven, so this first resolves the term via RxNorm."""
    source = "MedlinePlus Connect (NLM)"
    rx = await client.get(RXNORM_RXCUI, params={"name": term})
    rxcui = (rx.json().get("idGroup", {}).get("rxnormId") or [None])[0] if rx.status_code == 200 else None
    if not rxcui:
        return [], source, f'No RxNorm code for "{term}"; MedlinePlus Connect needs a code.'

    r = await client.get(MEDLINEPLUS_CONNECT_URL, params={
        "mainSearchCriteria.v.cs": RXNORM_OID,
        "mainSearchCriteria.v.c": rxcui,
        "knowledgeResponseType": "application/json",
    })
    if r.status_code != 200:
        return [], source, f"MedlinePlus Connect returned {r.status_code}."
    entries = (r.json().get("feed", {}) or {}).get("entry", []) or []
    if not entries:
        return [], source, f"No MedlinePlus Connect entry for RxCUI {rxcui}."

    cites = []
    for e in entries[: max(1, min(limit, 6))]:
        links = e.get("link", []) or []
        href = next((l.get("href") for l in links if l.get("href")), "https://medlineplus.gov/")
        cites.append({
            "source": f"MedlinePlus Connect · RxCUI {rxcui}",
            "source_kind": "patient-education",
            "title": _clean((e.get("title", {}) or {}).get("_value", "(untitled)"), 200),
            "url": href,
            "published": None,
            "snippet": _clean((e.get("summary", {}) or {}).get("_value", ""), 500)
                       or "Patient-education entry linked by drug code.",
            "_detail": {"rxcui": rxcui},
        })
    return cites, source, None
