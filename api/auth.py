"""Minimal session auth for the console and the doctor review portal.

An HMAC-signed cookie (`session`) carries {user, role, exp}. Two roles:
`admin` (the console + final review) and `doctor` (the /doctor approval portal).
No database, no external dependency. This is a local-tool gate, not production
identity — change the passwords and pin SESSION_SECRET before exposing the app.
"""
from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256

from .config import (
    ADMIN_PASSWORD,
    ADMIN_USER,
    DOCTOR_PASSWORD,
    DOCTOR_USER,
    SESSION_SECRET,
    SESSION_TTL,
)

COOKIE = "session"

_ACCOUNTS = {  # username -> (password, role)
    ADMIN_USER: (ADMIN_PASSWORD, "admin"),
    DOCTOR_USER: (DOCTOR_PASSWORD, "doctor"),
}


def check_credentials(username: str, password: str) -> str | None:
    """Return the role ('admin' | 'doctor') for valid credentials, else None."""
    entry = _ACCOUNTS.get((username or "").strip())
    if not entry:
        return None
    expected_pw, role = entry
    return role if hmac.compare_digest(password or "", expected_pw) else None


def _sig(body: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), body.encode(), sha256).hexdigest()[:32]


def issue_token(username: str, role: str = "admin") -> str:
    payload = {"u": username, "r": role, "exp": int(time.time()) + SESSION_TTL}
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"{body}.{_sig(body)}"


def verify_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sig(body)):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("exp", 0) < time.time():
        return None
    data.setdefault("r", "admin")  # tokens issued before roles existed
    return data


def cookie_max_age() -> int:
    return SESSION_TTL
