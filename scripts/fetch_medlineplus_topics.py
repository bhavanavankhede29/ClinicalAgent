"""Pull MedlinePlus's official bulk health-topic data file and stage each
English topic (title, full summary, synonyms, topic groups) as its own
document in the local knowledge base, then reindex.

This is NOT scraping — MedlinePlus publishes this XML file daily specifically
for bulk/programmatic use: https://medlineplus.gov/xml.html

    python scripts/fetch_medlineplus_topics.py
    python scripts/fetch_medlineplus_topics.py --no-reindex
    python scripts/fetch_medlineplus_topics.py --limit 200   # smaller test run
"""
from __future__ import annotations

import html
import io
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.config import KNOWLEDGE_DIR   # noqa: E402
from api.rag import reindex            # noqa: E402

BASE = "https://medlineplus.gov/xml"
_TAG_RE = re.compile(r"<[^>]+>")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "topic"


def _clean_html(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw or "")
    text = html.unescape(text)
    return " ".join(text.split())


def fetch_bytes() -> bytes:
    """The compressed file is published daily; try today, then walk back a
    few days in case today's hasn't landed yet."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for days_back in range(6):
            d = (date.today() - timedelta(days=days_back)).isoformat()
            url = f"{BASE}/mplus_topics_compressed_{d}.zip"
            r = client.get(url)
            if r.status_code == 200 and r.content:
                print(f"fetched {url} ({len(r.content):,} bytes)")
                return r.content
        raise RuntimeError("could not find a recent mplus_topics_compressed_*.zip")


def parse_topics(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
        with zf.open(xml_name) as f:
            root = ET.parse(f).getroot()
    for t in root.findall("health-topic"):
        if t.get("language") != "English":
            continue
        yield t


def write_topic(t: ET.Element, dest_dir: Path) -> None:
    title = t.get("title") or "Untitled topic"
    url = t.get("url") or ""
    topic_id = t.get("id") or "0"
    created = t.get("date-created") or ""

    also_called = [c.text.strip() for c in t.findall("also-called") if c.text and c.text.strip()]
    groups = [g.text.strip() for g in t.findall("group") if g.text and g.text.strip()]
    summary_el = t.find("full-summary")
    summary = _clean_html("".join(summary_el.itertext())) if summary_el is not None else ""

    lines = [
        "---",
        "source: MedlinePlus (National Library of Medicine)",
        f"title: {title}",
        f"published: {created}" if created else "published:",
        f"url: {url}",
        "jurisdiction: US",
        "category: health-topics",
        "---",
        "",
        f"# {title}",
        "",
    ]
    if also_called:
        lines += [f"Also called: {'; '.join(also_called)}.", ""]
    lines.append(summary or "_No summary text available for this topic._")
    if groups:
        lines += ["", f"Related MedlinePlus topic groups: {'; '.join(groups)}."]

    dest = dest_dir / f"{_slug(title)}-{topic_id}.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    zip_bytes = fetch_bytes()
    dest_dir = KNOWLEDGE_DIR / "health-topics"
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for t in parse_topics(zip_bytes):
        write_topic(t, dest_dir)
        count += 1
        if limit and count >= limit:
            break
    print(f"Staged {count} English MedlinePlus health-topic documents in {dest_dir}")

    if "--no-reindex" in sys.argv:
        print("skipped reindex (--no-reindex)")
        return
    stats = reindex()
    print("reindexed:", {k: stats[k] for k in ("documents", "chunks", "mode", "model") if k in stats})
    if stats.get("skipped"):
        print("skipped docs:", stats["skipped"])


if __name__ == "__main__":
    main()
