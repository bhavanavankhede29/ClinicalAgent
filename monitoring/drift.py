"""Request logging + input/output drift reporting.

`log_query()` appends one privacy-safe line per answered query to
monitoring/requests.jsonl:  timestamp, engine, mode, a salted hash of the query
(never the text), query length, #citations, source-kind mix, evidence-only flag,
KB-hit flag, latency. `api.main` calls it after every /api/query.

`python -m monitoring.drift report --baseline 30 --window 7` compares the last
`window` days against the preceding `baseline` days and flags shifts (population
stability index) in the output distributions — a signal to review / retrain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent / "requests.jsonl"
_SALT = os.environ.get("CLINICAL_AGENT_LOG_SALT", "clinical-agent")


def _qhash(text: str) -> str:
    return hashlib.sha256((_SALT + "|" + (text or "")).encode()).hexdigest()[:16]


def log_query(*, engine: str, req, resp, latency_ms: int) -> None:
    """Best-effort; never raises into the request path."""
    try:
        kinds = Counter(c.source_kind for c in resp.citations)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine": engine,
            "mode": req.mode,
            "q_hash": _qhash(req.query),
            "q_len": len(req.query or ""),
            "with_patient": bool(getattr(req, "fhir_patient_id", None)),
            "citations": len(resp.citations),
            "kinds": dict(kinds),
            "evidence_only": not resp.llm.used,
            "kb_hit": any(c.source_kind == "local-knowledge" for c in resp.citations),
            "trace_steps": len(resp.retrieval_trace),
            "latency_ms": latency_ms,
        }
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # pragma: no cover - monitoring must not break serving
        pass


# --- drift report ---------------------------------------------------------

def _load(days_from: float, days_to: float) -> list[dict]:
    if not LOG.exists():
        return []
    now = time.time()
    lo, hi = now - days_from * 86400, now - days_to * 86400
    out = []
    for ln in LOG.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        t = datetime.fromisoformat(r["ts"]).timestamp()
        if hi <= t <= lo:
            out.append(r)
    return out


def _dist(rows: list[dict], key) -> dict:
    c = Counter()
    for r in rows:
        c[key(r)] += 1
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def _psi(base: dict, cur: dict) -> float:
    keys = set(base) | set(cur)
    out = 0.0
    for k in keys:
        b = max(base.get(k, 0.0), 1e-6)
        c = max(cur.get(k, 0.0), 1e-6)
        out += (c - b) * math.log(c / b)
    return out


def report(baseline_days: float, window_days: float) -> int:
    base = _load(baseline_days + window_days, window_days)
    cur = _load(window_days, 0)
    if not base or not cur:
        print(f"not enough data (baseline={len(base)}, window={len(cur)} requests)")
        return 0

    print(f"baseline: {len(base)} requests over {baseline_days}d   "
          f"window: {len(cur)} over {window_days}d\n")
    metrics = {
        "mode": lambda r: r["mode"],
        "engine": lambda r: r["engine"],
        "evidence_only": lambda r: r["evidence_only"],
        "kb_hit": lambda r: r["kb_hit"],
        "citation_bucket": lambda r: "0" if r["citations"] == 0 else "1-5" if r["citations"] <= 5 else "6+",
    }
    flags = []
    for name, key in metrics.items():
        b, c = _dist(base, key), _dist(cur, key)
        psi = _psi(b, c)
        tag = "  OK" if psi < 0.1 else "  MINOR" if psi < 0.25 else "  DRIFT"
        if psi >= 0.25:
            flags.append(name)
        print(f"{name:16} PSI={psi:5.3f}{tag}")
        for k in sorted(set(b) | set(c)):
            print(f"    {str(k):14} {b.get(k,0):6.1%} -> {c.get(k,0):6.1%}")

    def _avg(rows, f):
        vals = [f(r) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\nlatency_ms  {int(_avg(base, lambda r: r['latency_ms']))} -> "
          f"{int(_avg(cur, lambda r: r['latency_ms']))}")
    print(f"citations   {_avg(base, lambda r: r['citations']):.1f} -> "
          f"{_avg(cur, lambda r: r['citations']):.1f}")

    if flags:
        print("\nDRIFT on: " + ", ".join(flags) + " — review inputs and model behaviour.")
        return 2
    print("\nno significant drift.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--baseline", type=float, default=30)
    r.add_argument("--window", type=float, default=7)
    sub.add_parser("count")
    args = ap.parse_args(argv)
    if args.cmd == "count":
        n = len(LOG.read_text(encoding="utf-8").splitlines()) if LOG.exists() else 0
        print(f"{n} logged requests in {LOG}")
        return 0
    return report(args.baseline, args.window)


if __name__ == "__main__":
    raise SystemExit(main())
