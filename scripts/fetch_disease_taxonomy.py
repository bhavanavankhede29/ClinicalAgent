"""Fetch a standard disease/condition taxonomy and stage it into the local
knowledge base (knowledge/guidelines/), so it gets chunked, embedded, and
becomes searchable via search_local_guidelines / the RAG index.

Source: NLM Clinical Table Search Service — "conditions" table. This is the
curated consumer/clinical condition-name list NLM uses to power MedlinePlus
and clinicaltrials.gov autocomplete. Free, no API key, no rate limit stated.
Docs: https://clinicaltables.nlm.nih.gov/apidoc/conditions/v3/doc.html

The API is search/autocomplete-shaped (substring match, capped at 500 results
per call, no offset param) rather than a bulk-export endpoint, so full
coverage is approximated by querying every two-letter substring ("aa".."zz",
676 calls) and de-duplicating the returned names. Run once; re-run any time
to refresh.

    python scripts/fetch_disease_taxonomy.py
"""
from __future__ import annotations

import asyncio
import string
import sys
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.config import KNOWLEDGE_DIR   # noqa: E402
from api.rag import reindex            # noqa: E402

API = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search"
PAIRS = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
CONCURRENCY = 15


async def _fetch_pair(client: httpx.AsyncClient, sem: asyncio.Semaphore, pair: str) -> list[str]:
    async with sem:
        try:
            r = await client.get(API, params={"terms": pair, "maxList": 500, "df": "primary_name"})
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        data = r.json()
        # Response shape is [total, codes[], extra, display[][]] but the extra
        # slot's presence/position varies by table — the display array is
        # reliably the last element.
        rows = data[-1] if data and isinstance(data[-1], list) else []
        return [row[0] for row in rows if row and row[0]]


async def collect() -> list[str]:
    seen: dict[str, str] = {}  # lowercase -> first-seen original casing
    async with httpx.AsyncClient(timeout=20.0) as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        results = await asyncio.gather(*(_fetch_pair(client, sem, p) for p in PAIRS))
    for names in results:
        for name in names:
            key = name.strip().lower()
            if key and key not in seen:
                seen[key] = name.strip()
    return sorted(seen.values(), key=str.lower)


def write_document(names: list[str]) -> Path:
    dest_dir = KNOWLEDGE_DIR / "guidelines"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "disease-condition-taxonomy-nlm.md"

    body = [
        "---",
        "source: NLM Clinical Table Search Service — Conditions table",
        "title: Unique Disease/Condition Name List (NLM Conditions Table)",
        f"published: {date.today().isoformat()}",
        "url: https://clinicaltables.nlm.nih.gov/apidoc/conditions/v3/doc.html",
        "jurisdiction: US",
        "category: guidelines",
        "---",
        "",
        "# Unique disease / condition name list",
        "",
        f"{len(names)} unique, de-duplicated condition names collected from the NLM "
        "Clinical Table Search Service's curated \"conditions\" table (the same "
        "consumer/clinical condition list that powers MedlinePlus and "
        "ClinicalTrials.gov autocomplete). This is a reference name list for "
        "matching/lookup, not clinical guidance — it carries no dosing, staging, "
        "or management content of its own.",
        "",
    ]
    body += [f"- {n}" for n in names]
    dest.write_text("\n".join(body) + "\n", encoding="utf-8")
    return dest


async def main() -> None:
    print(f"Querying {len(PAIRS)} two-letter substrings against {API} ...")
    names = await collect()
    print(f"Collected {len(names)} unique condition names.")
    dest = write_document(names)
    rel = dest.relative_to(KNOWLEDGE_DIR.parent)
    print(f"Wrote {rel} ({dest.stat().st_size:,} bytes)")

    if "--no-reindex" in sys.argv:
        print("skipped reindex (--no-reindex)")
        return
    stats = reindex()
    print("reindexed:", {k: stats[k] for k in ("documents", "chunks", "mode", "model") if k in stats})
    if stats.get("skipped"):
        print("skipped docs:", stats["skipped"])


if __name__ == "__main__":
    asyncio.run(main())
