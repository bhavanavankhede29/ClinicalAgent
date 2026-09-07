"""Load a document into the RAG knowledge corpus with provenance, then reindex.

    python -m api.kb_load "C:/downloads/ada-standards-2024.pdf" \
        --category guidelines \
        --source "American Diabetes Association" \
        --title "Standards of Care in Diabetes 2024" \
        --published 2024-01 --version 2024 --jurisdiction US \
        --url "https://diabetesjournals.org/care/issue/47/Supplement_1"

Copies the file into  knowledge/<category>/  and writes a  <name>.meta.json
sidecar so every retrieved passage can be cited with its source, date, version,
and jurisdiction. Markdown files may instead carry a `--- key: value ---`
front-matter block; the sidecar and front-matter are merged.

Supported: .pdf .md .markdown .txt .html .htm
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from .config import KNOWLEDGE_DIR
from .rag import _DOC_SUFFIXES, _META_KEYS, reindex

CATEGORIES = ("guidelines", "protocols", "drug-references", "reviews", "local")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "document"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="path to the document to load")
    ap.add_argument("--category", default="guidelines", choices=CATEGORIES)
    ap.add_argument("--source", help='e.g. "IDSA", "NICE", "hospital formulary committee"')
    ap.add_argument("--title", help="human title; defaults to the file name")
    ap.add_argument("--published", help="YYYY or YYYY-MM or YYYY-MM-DD")
    ap.add_argument("--revised", help="last-revised date, if different")
    ap.add_argument("--version", help='e.g. "2024", "v3.1"')
    ap.add_argument("--jurisdiction", help='e.g. "US", "UK", "EU", "hospital"')
    ap.add_argument("--url", help="canonical link to the source")
    ap.add_argument("--no-reindex", action="store_true", help="just stage the file")
    args = ap.parse_args(argv)

    src = Path(args.file).expanduser()
    if not src.is_file():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 2
    if src.suffix.lower() not in _DOC_SUFFIXES:
        print(f"error: unsupported type {src.suffix} (allowed: {sorted(_DOC_SUFFIXES)})", file=sys.stderr)
        return 2

    dest_dir = KNOWLEDGE_DIR / args.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(args.title or src.stem)
    dest = dest_dir / f"{stem}{src.suffix.lower()}"
    shutil.copyfile(src, dest)

    meta = {"category": args.category}
    for key in _META_KEYS:
        val = getattr(args, key.replace("-", "_"), None)
        if val:
            meta[key] = val
    meta.setdefault("title", args.title or src.stem)
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rel = dest.relative_to(KNOWLEDGE_DIR).as_posix()
    print(f"staged  {rel}")
    print(f"meta    {json.dumps(meta)}")

    if args.no_reindex:
        print("skipped reindex (--no-reindex)")
        return 0

    stats = reindex()
    print("reindexed:", json.dumps({k: stats[k] for k in
          ("documents", "chunks", "mode", "model") if k in stats}))
    if stats.get("skipped"):
        print("skipped docs:", stats["skipped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
