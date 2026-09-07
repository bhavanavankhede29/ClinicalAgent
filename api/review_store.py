"""File-backed store for generated treatment plans and their approval workflow.

    generated (pending_review)
        └─ doctor reviews ──approve──▶ approved ──admin acknowledges──▶ acknowledged
                           └─request_changes──▶ changes_requested

One JSON file (``REVIEW_STORE_PATH``), guarded by a process-wide lock. No
database; fine for a single-process local tool.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import REVIEW_STORE_PATH

_LOCK = threading.Lock()

STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_CHANGES = "changes_requested"
STATUS_REJECTED = "rejected"
STATUS_ACK = "acknowledged"

# What each role sees in its queue.
DOCTOR_QUEUE = (STATUS_PENDING, STATUS_CHANGES)
ADMIN_QUEUE = (STATUS_APPROVED, STATUS_ACK)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entry(role: str, actor: str, action: str, note: str = "") -> dict:
    return {"at": _now(), "role": role, "actor": actor, "action": action, "note": note or ""}


def _load() -> list[dict]:
    p = Path(REVIEW_STORE_PATH)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(rows: list[dict]) -> None:
    p = Path(REVIEW_STORE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(p)


def _summary(row: dict) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "created_by": row.get("created_by"),
        "patient_label": row.get("patient_label") or "(no patient linked)",
        "query": row.get("query", ""),
        "status": row.get("status"),
        "reviewed_by": row.get("reviewed_by"),
        "reviewed_at": row.get("reviewed_at"),
        "acknowledged_by": row.get("acknowledged_by"),
    }


def add_plan(*, query: str, answer_markdown: str, citations: list[dict],
             created_by: str, fhir_patient_id: str | None = None,
             patient_label: str | None = None) -> dict:
    row = {
        "id": uuid.uuid4().hex[:12],
        "created_at": _now(),
        "created_by": created_by,
        "query": query,
        "fhir_patient_id": fhir_patient_id,
        "patient_label": patient_label,
        "answer_markdown": answer_markdown,
        "citations": citations or [],
        "status": STATUS_PENDING,
        "doctor_notes": "",
        "decision": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "admin_response": "",
        "admin_responded_by": None,
        "admin_responded_at": None,
        "admin_note": "",
        "acknowledged_by": None,
        "acknowledged_at": None,
        "comments": [],
        "history": [_entry("admin", created_by, "submitted for review")],
    }
    with _LOCK:
        rows = _load()
        rows.insert(0, row)
        _save(rows)
    return row


def list_plans(*, statuses: tuple[str, ...] | None = None) -> list[dict]:
    with _LOCK:
        rows = _load()
    if statuses:
        rows = [r for r in rows if r.get("status") in statuses]
    return [_summary(r) for r in rows]


def get_plan(plan_id: str) -> dict | None:
    with _LOCK:
        for r in _load():
            if r["id"] == plan_id:
                return r
    return None


def doctor_review(plan_id: str, *, notes: str, decision: str, reviewer: str) -> dict | None:
    """decision: 'approve' -> approved, 'request_changes' -> changes_requested,
    'reject' -> rejected (terminal)."""
    new_status = {
        "approve": STATUS_APPROVED,
        "request_changes": STATUS_CHANGES,
        "reject": STATUS_REJECTED,
    }.get(decision, STATUS_CHANGES)
    action = {
        "approve": "approved",
        "request_changes": "requested changes",
        "reject": "rejected",
    }.get(decision, "requested changes")
    with _LOCK:
        rows = _load()
        for r in rows:
            if r["id"] == plan_id:
                r["doctor_notes"] = notes or ""
                r["decision"] = decision
                r["reviewed_by"] = reviewer
                r["reviewed_at"] = _now()
                r["status"] = new_status
                r.setdefault("history", []).append(_entry("doctor", reviewer, action, notes))
                _save(rows)
                return r
    return None


def admin_respond(plan_id: str, *, comment: str, admin: str) -> dict | None:
    """Admin replies to a change request and sends the plan back to the doctor."""
    with _LOCK:
        rows = _load()
        for r in rows:
            if r["id"] == plan_id:
                if r.get("status") != STATUS_CHANGES:
                    return {"_error": f"plan is '{r.get('status')}', not '{STATUS_CHANGES}'"}
                r["admin_response"] = comment or ""
                r["admin_responded_by"] = admin
                r["admin_responded_at"] = _now()
                r["status"] = STATUS_PENDING
                r.setdefault("history", []).append(
                    _entry("admin", admin, "responded, sent back to doctor", comment))
                _save(rows)
                return r
    return None


def add_comment(plan_id: str, *, text: str, by: str, role: str) -> dict | None:
    """Append a discussion comment (either role) to a plan's thread."""
    with _LOCK:
        rows = _load()
        for r in rows:
            if r["id"] == plan_id:
                r.setdefault("comments", []).append({
                    "id": uuid.uuid4().hex[:8], "at": _now(), "by": by, "role": role,
                    "text": text or "", "edited_at": None,
                })
                _save(rows)
                return r
    return None


def edit_comment(plan_id: str, comment_id: str, *, text: str, by: str) -> dict | None:
    """Update a comment — only its original author may."""
    with _LOCK:
        rows = _load()
        for r in rows:
            if r["id"] == plan_id:
                for c in r.get("comments", []):
                    if c["id"] == comment_id:
                        if c["by"] != by:
                            return {"_error": "you can only edit your own comment"}
                        c["text"] = text or ""
                        c["edited_at"] = _now()
                        _save(rows)
                        return r
                return {"_error": "comment not found"}
    return None


def doctor_recent_reviews(reviewer: str, *, hours: int = 24) -> list[dict]:
    """Every review action this doctor took in the last `hours`, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: list[dict] = []
    with _LOCK:
        rows = _load()
    for r in rows:
        for h in r.get("history", []):
            if h.get("role") != "doctor" or h.get("actor") != reviewer:
                continue
            try:
                when = datetime.fromisoformat(h["at"])
            except (ValueError, KeyError, TypeError):
                continue
            if when >= cutoff:
                out.append({
                    "plan_id": r["id"], "patient_label": r.get("patient_label"),
                    "query": r.get("query", ""), "action": h.get("action", ""),
                    "note": h.get("note", ""), "at": h.get("at"),
                    "current_status": r.get("status"),
                })
    out.sort(key=lambda x: x["at"] or "", reverse=True)
    return out


def admin_acknowledge(plan_id: str, *, note: str, admin: str) -> dict | None:
    with _LOCK:
        rows = _load()
        for r in rows:
            if r["id"] == plan_id:
                if r.get("status") != STATUS_APPROVED:
                    return {"_error": f"plan is '{r.get('status')}', not '{STATUS_APPROVED}'"}
                r["admin_note"] = note or ""
                r["acknowledged_by"] = admin
                r["acknowledged_at"] = _now()
                r["status"] = STATUS_ACK
                r.setdefault("history", []).append(_entry("admin", admin, "acknowledged & finalised", note))
                _save(rows)
                return r
    return None
