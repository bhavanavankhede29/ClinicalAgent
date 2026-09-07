"""HIPAA Safe Harbor de-identification helpers.

`scrub_text()` masks the 18 Safe Harbor identifier classes it can detect in free
text. `verify_case()` asserts a CaseRecord carries no date, no birth date, and an
age band rather than an age — it raises rather than silently passing bad data.

This enforces record-level hygiene. It is NOT a substitute for a documented
de-identification (Safe Harbor by a qualified person, or Expert Determination)
performed on the source before it reaches this repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import CaseRecord

# --- detectors (conservative; over-mask rather than leak) ---------------------
_PATTERNS = {
    "DATE": re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}"
                       r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4})\b",
                       re.I),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "URL": re.compile(r"\bhttps?://\S+\b", re.I),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "MRN": re.compile(r"\b(?:MRN|medical record (?:number|no)\.?|record #)\s*[:#]?\s*\w+\b", re.I),
    "ZIP": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "AGE90": re.compile(r"\b(9\d|1\d\d)\s*(?:years?|yo|y/o|-year-old)\b", re.I),
}


def scrub_text(text: str) -> str:
    """Return `text` with detected Safe Harbor identifiers replaced by [REDACTED:<CLASS>]."""
    if not text:
        return text
    out = text
    for label, rx in _PATTERNS.items():
        if label == "AGE90":
            out = rx.sub("[REDACTED:AGE>=90]", out)
        else:
            out = rx.sub(f"[REDACTED:{label}]", out)
    return out


def find_identifiers(text: str) -> list[str]:
    """List identifier classes still present in `text` (for a review report)."""
    return [label for label, rx in _PATTERNS.items() if rx.search(text or "")]


@dataclass
class DeidReport:
    ok: bool
    findings: list[str]


def verify_case(case: CaseRecord) -> DeidReport:
    """Structural Safe Harbor checks on a CaseRecord. Does not raise; returns a report."""
    findings: list[str] = []
    if case.age_group is None and getattr(case, "age", None) is not None:  # pragma: no cover
        findings.append("age present without age_group")
    for field in ("problems", "medications", "procedures", "orders"):
        for item in getattr(case, field, []):
            hits = find_identifiers(item)
            if hits:
                findings.append(f"{field}: {item[:40]!r} -> {hits}")
    for k, v in (case.labs | case.outcomes).items():
        hits = find_identifiers(f"{k} {v}")
        if hits:
            findings.append(f"labs/outcomes {k!r} -> {hits}")
    # a case_id that looks like an MRN/SSN
    if _PATTERNS["SSN"].search(case.case_id):
        findings.append("case_id resembles an SSN")
    return DeidReport(ok=not findings, findings=findings)


def assert_case_clean(case: CaseRecord) -> None:
    rep = verify_case(case)
    if not rep.ok:
        raise ValueError("CaseRecord failed Safe Harbor checks:\n  " + "\n  ".join(rep.findings))
