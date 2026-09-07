"""Local knowledge base — the RAG layer.

The live APIs (PubMed, openFDA, ClinicalTrials.gov, RxNorm, MedlinePlus, …) cover
published evidence. This module covers the documents an institution holds that
are *not* on any public API: local protocols, formulary notes, order-set
rationale, guideline excerpts.

Pipeline
--------
1. Load ``*.md`` / ``*.txt`` files from ``knowledge/``.
2. Chunk each file into **500-character windows with 100 characters of overlap**
   (config: ``RAG_CHUNK_CHARS`` / ``RAG_CHUNK_OVERLAP``), keeping the nearest
   Markdown heading as the chunk label.
3. **Embed** every chunk:
     * ``VOYAGE_API_KEY`` set  -> dense vectors from Voyage AI (``EMBEDDING_MODEL``),
       retrieval is cosine similarity over the embeddings.
     * no key                 -> local TF-IDF vector, cosine over that (works
       offline, no dependencies).
4. Persist to ``knowledge/.kb_index.json``.
5. At query time the same embedding is applied to the question; the top chunks
   are returned as citations and handed to the LLM as context — the agent calls
   ``search_local_guidelines`` first on every query.

Rebuild: ``python -m api.rag``  or  ``POST /api/rag/reindex``.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

try:
    from pypdf import PdfReader
    _PDF_OK = True
except ModuleNotFoundError:  # optional: PDFs skipped with a note if absent
    _PDF_OK = False

import httpx

from .config import (
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    RAG_CHUNK_CHARS,
    RAG_CHUNK_OVERLAP,
    RAG_INDEX_PATH,
    VOYAGE_API_KEY,
    VOYAGE_BASE_URL,
)

INDEX_FORMAT = 4

_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
_DOC_SUFFIXES = _TEXT_SUFFIXES | {".pdf", ".html", ".htm"}
_META_KEYS = ("source", "title", "published", "revised", "version", "jurisdiction", "url", "category")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_MD_NOISE_RE = re.compile(r"[*_`>|#]+")
_TAG_RE = re.compile(r"<[^>]+>")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.S)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "as", "at", "by", "it", "this", "that", "from", "was", "were",
    "if", "then", "than", "not", "no", "any", "all", "may", "can", "should",
    "which", "who", "when", "where", "into", "per", "via",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) < 40]


def _clean(text: str, limit: int = 600) -> str:
    text = " ".join(_MD_NOISE_RE.sub(" ", text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_document(path: Path) -> tuple[str, dict]:
    """Return (plain_text, provenance_meta) for a corpus file.

    Provenance comes from a `<name>.meta.json` sidecar and/or a leading
    `--- key: value ---` front-matter block in a Markdown/text file.
    """
    suffix = path.suffix.lower()
    meta: dict = {}

    sidecar = path.with_suffix(".meta.json")
    if sidecar.is_file():
        try:
            meta.update({k: v for k, v in json.loads(sidecar.read_text("utf-8")).items()
                         if k in _META_KEYS and v})
        except (json.JSONDecodeError, OSError):
            pass

    if suffix == ".pdf":
        if not _PDF_OK:
            return "", {**meta, "_error": "pypdf not installed"}
        try:
            reader = PdfReader(str(path))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            info = reader.metadata or {}
            meta.setdefault("title", (info.get("/Title") or "").strip() or path.stem)
        except Exception as exc:  # corrupt / encrypted PDF
            return "", {**meta, "_error": f"PDF read failed: {exc}"}
        return text, meta

    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix in {".html", ".htm"}:
        raw = _TAG_RE.sub(" ", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw))
        return raw, meta

    fm = _FRONTMATTER_RE.match(raw)
    if fm:
        for line in fm.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip().lower(), v.strip().strip('"').strip("'")
                if k in _META_KEYS and v:
                    meta[k] = v
        raw = raw[fm.end():]
    return raw, meta


def _provenance(meta: dict) -> str:
    bits = [meta.get("source"), meta.get("version"),
            meta.get("revised") or meta.get("published"), meta.get("jurisdiction")]
    return " · ".join(b for b in bits if b)


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# --- chunking ---------------------------------------------------------------

def _sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings into (heading, body) pairs."""
    heading, buf, out = "", [], []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf:
                out.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = m.group(1).strip()
        else:
            buf.append(line)
    if buf:
        out.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in out if b]


def _chunk_document(text: str,
                    size: int = RAG_CHUNK_CHARS,
                    overlap: int = RAG_CHUNK_OVERLAP) -> list[dict]:
    """Sliding-window chunks: `size` chars, stepping `size - overlap` each time,
    never crossing a heading boundary."""
    step = max(1, size - overlap)
    chunks: list[dict] = []
    for heading, body in _sections(text):
        body = " ".join(body.split())
        if not body:
            continue
        if len(body) <= size:
            chunks.append({"heading": heading, "text": body})
            continue
        start = 0
        while start < len(body):
            chunks.append({"heading": heading, "text": body[start:start + size]})
            if start + size >= len(body):
                break
            start += step
    return chunks


# --- embedding backends ----------------------------------------------------

def _voyage_embed_sync(texts: list[str], input_type: str, batch: int = 64) -> list[list[float]]:
    """Batch-embed with Voyage AI, retrying on 429 / 5xx with backoff so a large
    reindex survives the free-tier rate limit."""
    import time as _time

    vectors: list[list[float]] = []
    with httpx.Client(timeout=90.0) as client:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            for attempt in range(3):
                r = client.post(
                    f"{VOYAGE_BASE_URL}/embeddings",
                    headers={"authorization": f"Bearer {VOYAGE_API_KEY}",
                             "content-type": "application/json"},
                    json={"input": chunk, "model": EMBEDDING_MODEL, "input_type": input_type},
                )
                if r.status_code in (429, 500, 502, 503) and attempt < 2:
                    wait = float(r.headers.get("retry-after", 0)) or (5 * (attempt + 1))
                    _time.sleep(min(wait, 15))
                    continue
                r.raise_for_status()
                break
            rows = sorted(r.json()["data"], key=lambda d: d["index"])
            vectors.extend(_unit(row["embedding"]) for row in rows)
            _time.sleep(1.0)  # stay under the requests-per-minute cap
    return vectors


async def _voyage_embed_query(client: httpx.AsyncClient, text: str) -> list[float]:
    r = await client.post(
        f"{VOYAGE_BASE_URL}/embeddings",
        headers={"authorization": f"Bearer {VOYAGE_API_KEY}", "content-type": "application/json"},
        json={"input": [text], "model": EMBEDDING_MODEL, "input_type": "query"},
        timeout=30.0,
    )
    r.raise_for_status()
    return _unit(r.json()["data"][0]["embedding"])


# --- index ---------------------------------------------------------------

class KnowledgeIndex:
    def __init__(self) -> None:
        self.format = INDEX_FORMAT
        self.mode = "tfidf"            # "voyage" | "tfidf"
        self.model: str | None = None
        self.dim: int | None = None
        self.built_at: str | None = None
        self.docs: list[str] = []
        self.doc_meta: dict[str, dict] = {}   # doc name -> provenance
        self.chunks: list[dict] = []   # {id, doc, source_path, ordinal, heading, text}
        self.vectors: list = []        # list[list[float]] (voyage) | list[dict] (tfidf)
        self.df: Counter = Counter()   # tfidf only

    # ---- build --------------------------------------------------------

    def build_from_dir(self, directory: Path) -> dict:
        directory = Path(directory)
        self.__init__()
        if not directory.exists():
            self.built_at = _now_iso()
            return {"documents": 0, "chunks": 0, "mode": "tfidf",
                    "note": f"{directory} does not exist"}

        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file()
            and p.suffix.lower() in _DOC_SUFFIXES
            and not p.name.startswith(".")
            and p.name.lower() != "readme.md"
            and not p.name.lower().endswith(".meta.json")
        )
        raw: list[dict] = []
        skipped: list[str] = []
        for path in files:
            doc = path.relative_to(directory).as_posix()
            text, meta = _read_document(path)
            if not meta.get("category") and "/" in doc:
                meta["category"] = doc.split("/", 1)[0]
            if meta.get("_error") or not text.strip():
                skipped.append(f"{doc} ({meta.get('_error', 'no extractable text')})")
                continue
            self.docs.append(doc)
            self.doc_meta[doc] = {k: v for k, v in meta.items() if not k.startswith("_")}
            for ordinal, ch in enumerate(_chunk_document(text)):
                raw.append({
                    "id": len(raw), "doc": doc, "source_path": str(path),
                    "ordinal": ordinal, "heading": ch["heading"], "text": ch["text"],
                })
        self.chunks = raw
        texts = [c["text"] for c in raw]
        embed_note = None

        if VOYAGE_API_KEY and texts:
            try:
                self.vectors = _voyage_embed_sync(texts, "document")
                self.mode = "voyage"
                self.model = EMBEDDING_MODEL
                self.dim = len(self.vectors[0]) if self.vectors else None
            except httpx.HTTPStatusError as exc:
                embed_note = (f"Voyage embedding failed ({exc.response.status_code}); "
                              "built a TF-IDF index instead. Re-run reindex to try again.")

        if self.mode != "voyage":
            tokenized = [_tokenize(t) for t in texts]
            for toks in tokenized:
                self.df.update(set(toks))
            n = len(raw)
            self.vectors = [self._tfidf_vec(toks, n) for toks in tokenized]
            self.mode = "tfidf"
            self.model = None

        self.built_at = _now_iso()
        stats = {"documents": len(self.docs), "chunks": len(raw),
                 "mode": self.mode, "model": self.model, "dim": self.dim}
        if skipped:
            stats["skipped"] = skipped
        if embed_note:
            stats["note"] = embed_note
        return stats

    # ---- tfidf helpers ----------------------------------------------

    def _idf(self, term: str, n: int) -> float:
        return math.log((n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def _tfidf_vec(self, tokens: list[str], n: int) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec = {t: (1.0 + math.log(c)) * self._idf(t, n) for t, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    # ---- query ----------------------------------------------------

    def search_tfidf(self, query: str, k: int = 4) -> list[tuple[dict, float]]:
        if not self.chunks:
            return []
        q = self._tfidf_vec(_tokenize(query), len(self.chunks))
        if not q:
            return []
        scored = [
            (chunk, sum(w * vec.get(t, 0.0) for t, w in q.items()))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored = [s for s in scored if s[1] > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, min(k, 8))]

    def search_dense(self, query_vec: list[float], k: int = 4) -> list[tuple[dict, float]]:
        if not self.chunks or not query_vec:
            return []
        scored = [
            (chunk, sum(a * b for a, b in zip(query_vec, vec)))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, min(k, 8))]

    # ---- persistence -------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": self.format, "mode": self.mode, "model": self.model, "dim": self.dim,
            "built_at": self.built_at, "docs": self.docs, "doc_meta": self.doc_meta,
            "df": dict(self.df), "chunks": self.chunks, "vectors": self.vectors,
        }), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.format = data.get("format", 0)
        self.mode = data.get("mode", "tfidf")
        self.model = data.get("model")
        self.dim = data.get("dim")
        self.built_at = data.get("built_at")
        self.docs = data.get("docs", [])
        self.doc_meta = data.get("doc_meta", {})
        self.df = Counter(data.get("df", {}))
        self.chunks = data.get("chunks", [])
        vecs = data.get("vectors", [])
        if self.mode == "voyage":
            self.vectors = [[float(x) for x in v] for v in vecs]
        else:
            self.vectors = [{t: float(w) for t, w in v.items()} for v in vecs]

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def stale(self) -> bool:
        """The on-disk index must be rebuilt. Switching embedding backend
        (tfidf <-> voyage) is a deliberate `reindex()`, not an automatic
        rebuild on every startup — so a valid index of either mode is kept."""
        return (self.format != INDEX_FORMAT
                or (self.mode == "voyage" and self.model != EMBEDDING_MODEL))


# --- module singleton --------------------------------------------------------

_INDEX: KnowledgeIndex | None = None


def get_index() -> KnowledgeIndex:
    global _INDEX
    if _INDEX is None:
        idx = KnowledgeIndex()
        if Path(RAG_INDEX_PATH).exists():
            try:
                idx.load(RAG_INDEX_PATH)
            except (json.JSONDecodeError, OSError):
                idx = KnowledgeIndex()
        if idx.n_chunks == 0 or idx.stale:
            try:
                idx.build_from_dir(KNOWLEDGE_DIR)
                idx.save(RAG_INDEX_PATH)
            except (OSError, httpx.HTTPError):
                pass
        _INDEX = idx
    return _INDEX


def reindex() -> dict:
    """Rebuild the index from the corpus directory and swap it in."""
    global _INDEX
    idx = KnowledgeIndex()
    stats = idx.build_from_dir(KNOWLEDGE_DIR)
    try:
        idx.save(RAG_INDEX_PATH)
    except OSError as exc:  # pragma: no cover
        stats["save_error"] = str(exc)
    _INDEX = idx
    return {**stats, "built_at": idx.built_at,
            "chunk_chars": RAG_CHUNK_CHARS, "chunk_overlap": RAG_CHUNK_OVERLAP,
            "index_path": str(RAG_INDEX_PATH)}


async def search_knowledge_base(client: httpx.AsyncClient | None = None, *, query: str, k: int = 4):
    """Retrieval function in the standard (raw_citations, source_api, note) shape.

    Uses the passed AsyncClient (if any) to embed the query when the index is a
    dense Voyage index; creates a short-lived client otherwise.
    """
    idx = get_index()
    source_api = (f"Local knowledge base ({idx.mode}"
                  + (f":{idx.model}" if idx.model else "") + " cosine)")
    if idx.n_chunks == 0:
        return [], source_api, "Knowledge base is empty. Add files under knowledge/ and reindex."

    if idx.mode == "voyage":
        try:
            if client is not None:
                qvec = await _voyage_embed_query(client, query)
            else:
                async with httpx.AsyncClient() as c:
                    qvec = await _voyage_embed_query(c, query)
        except httpx.HTTPError as exc:
            return [], source_api, f"Query embedding failed: {exc}"
        hits = idx.search_dense(qvec, k)
    else:
        hits = idx.search_tfidf(query, k)

    if not hits:
        return [], source_api, "No local passage matched the query."

    cites = []
    for chunk, score in hits:
        meta = idx.doc_meta.get(chunk["doc"], {})
        label = meta.get("title") or chunk["doc"]
        where = label + (f" › {chunk['heading']}" if chunk["heading"]
                         else f" › chunk {chunk['ordinal']}")
        prov = _provenance(meta)
        doc_url = meta.get("url") or f"/api/kb/doc/{quote(chunk['doc'])}?chunk={chunk['ordinal']}"
        if not meta.get("url") and chunk["heading"]:
            doc_url += "#" + quote(chunk["heading"].lower().replace(" ", "-"))
        cites.append({
            "source": f"Local KB · {where}" + (f"  ({prov})" if prov else ""),
            "source_kind": "local-knowledge",
            "title": chunk["heading"] or label,
            "url": doc_url,
            "published": meta.get("revised") or meta.get("published") or idx.built_at,
            "snippet": _clean(chunk["text"], 600),
            "_detail": {
                "file": chunk["source_path"],
                "document": chunk["doc"],
                "heading": chunk["heading"],
                "chunk_ordinal": chunk["ordinal"],
                "similarity": round(float(score), 4),
                "retrieval": idx.mode,
                "provenance": meta,
                "full_text": chunk["text"],
            },
        })
    return cites, source_api, None


if __name__ == "__main__":
    print(json.dumps(reindex(), indent=2))
