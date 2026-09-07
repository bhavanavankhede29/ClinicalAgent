"""Sandbox FHIR R4 client.

Reads synthetic-patient data from a public FHIR server (``FHIR_BASE_URL``,
default the HAPI test server) so a query can be grounded in a demo patient's
problem list, active medications, recent results, and allergies.

Synthetic data only. This is decision-support context, never a substitute for
the chart, and the server must never be pointed at real PHI.
"""
from __future__ import annotations

import asyncio

import httpx

from .config import FHIR_BASE_URL
from .net import cached_call, get_client

_HEADERS = {"Accept": "application/fhir+json"}


def _text(concept: dict | None) -> str:
    if not concept:
        return ""
    if concept.get("text"):
        return concept["text"]
    for coding in concept.get("coding", []) or []:
        if coding.get("display"):
            return coding["display"]
        if coding.get("code"):
            return coding["code"]
    return ""


def _status(res: dict, key: str) -> str:
    return (res.get(key, {}) or {}).get("coding", [{}])[0].get("code", "") or ""


_SYMPTOM_HINTS = (
    "pain", "ache", "cough", "fever", "nausea", "vomit", "diarrh", "fatigue", "malaise",
    "headache", "dizz", "dyspnea", "dyspnoea", "breath", "rash", "itch", "pruritus",
    "swelling", "edema", "oedema", "sprain", "strain", "sore throat", "congestion",
    "sneez", "chill", "weakness", "numb", "tingling", "palpitation", "syncope",
    "bleeding", "discharge", "cramp", "spasm", "stiffness", "insomnia", "wheez",
    "blurred vision", "tinnitus", "kreuzschmerz", "(finding)",
)
_NOT_SYMPTOM = (
    "typhoid fever", "yellow fever", "rheumatic fever", "scarlet fever", "dengue fever",
    "q fever", "lassa fever", "glandular fever", "hay fever", "spotted fever",
)


def _is_symptom(name: str, verification: str = "") -> bool:
    """Heuristic split for messy sandbox data: a Condition whose display names a
    finding/symptom rather than a disease. Keyword-based — verificationStatus in
    this public data is too unreliable to use."""
    low = name.lower()
    if "(disorder)" in low or any(x in low for x in _NOT_SYMPTOM):
        return False
    return any(h in low for h in _SYMPTOM_HINTS)


def _name(patient: dict) -> str:
    names = patient.get("name") or []
    if not names:
        return f"Patient {patient.get('id', '?')}"
    n = names[0]
    if n.get("text"):
        return n["text"]
    return " ".join(list(n.get("given", [])) + [n.get("family", "")]).strip() or f"Patient {patient.get('id', '?')}"


async def _bundle(client: httpx.AsyncClient, path: str, params: dict) -> list[dict]:
    r = await client.get(f"{FHIR_BASE_URL}/{path}", params=params, headers=_HEADERS)
    r.raise_for_status()
    return [e.get("resource", {}) for e in (r.json().get("entry") or [])]


# The public HAPI test server is full of throw-away demo/test uploads. On a plain
# browse these crowd out the real synthetic (Synthea) patients, so drop the
# obvious ones and don't sort by -_lastUpdated (which surfaces exactly the junk).
_JUNK_NAME_BITS = (
    "demo", "test", "poctest", "sample", "example", "dummy", "fixture", "xxx",
    "asdf", "qwer", "delete", "temp ", "foobar", "langcare", "burnout-demo",
)


def _looks_like_junk(display: str) -> bool:
    low = display.lower()
    if not any(ch.isalpha() for ch in low.replace("patient", "")):
        return True
    if any(bit in low for bit in _JUNK_NAME_BITS):
        return True
    # slug-style names: "Patient burnout-demo-patient-m-alvarez-2026-09-05-52"
    if low.startswith("patient ") and ("-" in low or any(ch.isdigit() for ch in low)):
        return True
    return False


async def search_patients(name: str | None = None, count: int = 100) -> list[dict]:
    """Search patients by name; with no name, return a browse list of the
    server's synthetic patients with the throw-away demo/test uploads removed."""
    name = (name or "").strip()
    count = max(1, min(count, 200))

    async def _run():
        client = get_client()
        want = count if name else 200          # over-fetch on browse to survive filtering
        params = {"_count": want}
        if name:
            params["name"] = name
        else:
            params["_sort"] = "_id"            # stable order; NOT -_lastUpdated (junk magnet)
        rows = await _bundle(client, "Patient", params)
        out, seen = [], set()
        for p in rows:
            if p.get("resourceType") != "Patient" or not p.get("id"):
                continue
            display = _name(p)
            if not name:
                if _looks_like_junk(display):
                    continue
                key = (display.lower().strip(), p.get("birthDate") or "")
                if key in seen:
                    continue
                seen.add(key)
            out.append({
                "id": p.get("id"),
                "name": display,
                "gender": p.get("gender"),
                "birth_date": p.get("birthDate"),
            })
            if len(out) >= count:
                break
        return out

    return await cached_call("fhir_patient_search", {"name": name, "count": count}, _run)


async def patient_context(patient_id: str) -> dict:
    """Fetch a compact clinical picture for one patient plus a plain-text summary."""

    async def _run():
        client = get_client()
        pr = await client.get(f"{FHIR_BASE_URL}/Patient/{patient_id}", headers=_HEADERS)
        pr.raise_for_status()
        patient = pr.json()

        conditions, meds, obs, allergies = await asyncio.gather(
            _bundle(client, "Condition", {"patient": patient_id, "_count": 50}),
            _bundle(client, "MedicationRequest", {"patient": patient_id, "_count": 50}),
            _bundle(client, "Observation", {"patient": patient_id, "_count": 15, "_sort": "-date"}),
            _bundle(client, "AllergyIntolerance", {"patient": patient_id, "_count": 30}),
        )

        problems = []
        for c in conditions:
            if c.get("resourceType") != "Condition":
                continue
            name = _text(c.get("code"))
            if not name:
                continue
            verification = _status(c, "verificationStatus")
            problems.append({
                "name": name,
                "status": _status(c, "clinicalStatus"),
                "verification": verification,
                "onset": c.get("onsetDateTime"),
                "kind": "symptom" if _is_symptom(name, verification) else "condition",
            })
        symptoms = [p for p in problems if p["kind"] == "symptom"]
        conditions_only = [p for p in problems if p["kind"] == "condition"]
        medications = [
            {"name": _text(m.get("medicationCodeableConcept")),
             "status": m.get("status"),
             "dosage": (m.get("dosageInstruction") or [{}])[0].get("text"),
             "authored": m.get("authoredOn")}
            for m in meds if m.get("resourceType") == "MedicationRequest"
            and _text(m.get("medicationCodeableConcept"))
        ]
        results = []
        for o in obs:
            if o.get("resourceType") != "Observation":
                continue
            vq = o.get("valueQuantity") or {}
            value = (f"{vq.get('value')} {vq.get('unit', '')}".strip()
                     if vq else o.get("valueString") or _text(o.get("valueCodeableConcept")))
            results.append({"name": _text(o.get("code")), "value": value,
                            "date": o.get("effectiveDateTime")})
        allergy_list = [
            {"substance": _text(a.get("code")),
             "criticality": a.get("criticality"),
             "reaction": _text(((a.get("reaction") or [{}])[0].get("manifestation") or [{}])[0])}
            for a in allergies if a.get("resourceType") == "AllergyIntolerance" and _text(a.get("code"))
        ]

        _active = ("", "active", "recurrence", "relapse")
        active_conditions = [p["name"] for p in conditions_only if p["status"] in _active]
        active_symptoms = [p["name"] for p in symptoms if p["status"] in _active]
        active_meds = [m["name"] for m in medications if m["status"] in ("", "active")]

        parts = [f"Patient: {_name(patient)}"
                 f"{', ' + patient.get('gender') if patient.get('gender') else ''}"
                 f"{', DOB ' + patient['birthDate'] if patient.get('birthDate') else ''}."]
        if active_conditions:
            parts.append("Conditions: " + "; ".join(active_conditions[:15]) + ".")
        if active_symptoms:
            parts.append("Symptoms: " + "; ".join(active_symptoms[:15]) + ".")
        if active_meds:
            parts.append("Current medications: " + "; ".join(active_meds[:20]) + ".")
        if allergy_list:
            parts.append("Allergies: " + "; ".join(a["substance"] for a in allergy_list[:10]) + ".")
        if results:
            recent = [f"{r['name']} {r['value']}".strip() for r in results[:8] if r["name"]]
            if recent:
                parts.append("Recent results: " + "; ".join(recent) + ".")
        parts.append("(Synthetic FHIR sandbox data — verify against the chart.)")

        return {
            "patient": {"id": patient.get("id"), "name": _name(patient),
                        "gender": patient.get("gender"), "birth_date": patient.get("birthDate")},
            "problems": problems,          # all conditions (with a "kind" field)
            "conditions": conditions_only,
            "symptoms": symptoms,
            "medications": medications,
            "results": results,
            "allergies": allergy_list,
            "has_data": bool(conditions_only or symptoms or medications or results or allergy_list),
            "summary": " ".join(parts),
            "source": f"FHIR sandbox · {FHIR_BASE_URL}",
        }

    return await cached_call("fhir_patient_context", {"id": patient_id}, _run)


async def list_patients_with_data(name: str | None = None, count: int = 40) -> list[dict]:
    """Like search_patients, but scores each candidate's chart and returns the
    richest first — patients with real clinical content (conditions, meds,
    allergies) ahead of those carrying only a stray vital sign — so a clinician
    lands on a usable record, not a near-empty one. Each row gains a
    ``chart`` count dict and a short ``chart_summary`` string for the UI."""
    scan = min(max(count * 4, 80), 200)
    candidates = await search_patients(name, count=scan)
    sem = asyncio.Semaphore(10)

    async def _check(p: dict) -> dict | None:
        async with sem:
            try:
                ctx = await patient_context(p["id"])
            except Exception:  # noqa: BLE001 — a flaky sandbox record just gets skipped
                return None
        counts = {
            "conditions": len(ctx.get("conditions") or []),
            "symptoms": len(ctx.get("symptoms") or []),
            "medications": len(ctx.get("medications") or []),
            "results": len(ctx.get("results") or []),
            "allergies": len(ctx.get("allergies") or []),
        }
        if not any(counts.values()):
            return None
        # Weight clinically meaningful content over lone observations.
        score = (3 * counts["conditions"] + 3 * counts["medications"]
                 + 2 * counts["allergies"] + counts["symptoms"] + counts["results"])
        clinical = bool(counts["conditions"] or counts["medications"] or counts["allergies"])
        parts = [f"{n} {label}" for label, n in (
            ("cond", counts["conditions"]), ("sympt", counts["symptoms"]),
            ("meds", counts["medications"]), ("results", counts["results"]),
            ("allergies", counts["allergies"])) if n]
        return {**p, "chart": counts, "chart_summary": " · ".join(parts),
                "_score": score, "_clinical": clinical}

    checked = [p for p in await asyncio.gather(*(_check(p) for p in candidates)) if p]
    # Rich clinical charts first, then observation-only ones; richest within each.
    checked.sort(key=lambda p: (p["_clinical"], p["_score"]), reverse=True)
    for p in checked:
        p.pop("_score", None)
        p.pop("_clinical", None)
    if checked:
        return checked[:count]

    # The sandbox sometimes has no charted patients in the browse window at all
    # (a flood of empty demo uploads). Rather than a blank panel, hand back the
    # raw candidates so the clinician can still pick one or search by name.
    for c in candidates:
        c["chart_summary"] = "no charted data — search by name or use Manual entry"
    return candidates[:count]
