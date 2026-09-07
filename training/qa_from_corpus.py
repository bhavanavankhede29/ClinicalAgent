"""Guideline-derived Q&A pairs from the RAG corpus.

Each indexed passage (a guideline recommendation, a drug-label section) is
decomposed by the LLM into clinically useful question/answer pairs, each grounded
in and cited to that passage. Output is GuidelineQA records plus, optionally, the
same content as SFTExample records for instruction tuning.

    python -m training.qa_from_corpus --limit 40 --category guidelines \
        --out data/guideline_qa.jsonl --sft-out data/sft_corpus.jsonl

Nothing here is training-ready until a clinician sets reviewed=true (see
training/review.py).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import datetime, timezone

from api.rag import get_index

from .schemas import Citation, GuidelineQA, Provenance, SFTExample, SYSTEM_PROMPT, write_jsonl
from ._llm import MODEL, ask, available, json_array

_SYS = (
    "You convert a single clinical guideline or drug-label passage into training "
    "Q&A. Rules: every answer must be fully supported by the passage provided — "
    "never add outside facts; write answers as evidence statements, not directives; "
    "keep each answer 1-4 sentences. If the passage is boilerplate with no clinical "
    "content, return an empty array."
)


def _prompt(passage: str, doc_title: str) -> str:
    return (
        f"Source document: {doc_title}\n\nPassage:\n\"\"\"\n{passage}\n\"\"\"\n\n"
        "Return a JSON array of up to 3 objects: "
        '{"question": "...", "answer": "...", '
        '"recommendation_class": "<if stated, else null>", '
        '"evidence_level": "<if stated, else null>"}'
    )


def _cid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


async def build(limit: int, category: str | None, out: str, sft_out: str | None) -> None:
    if not available():
        raise SystemExit("ANTHROPIC_API_KEY not set — cannot generate Q&A.")

    idx = get_index()
    chunks = [c for c in idx.chunks
              if not category or c["doc"].startswith(category.rstrip("/") + "/")]
    if not chunks:
        raise SystemExit(f"No indexed chunks{f' under {category}/' if category else ''}. "
                         "Load documents and run `python -m api.rag` first.")
    chunks = chunks[:limit]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    qa_records: list[GuidelineQA] = []
    sft_records: list[SFTExample] = []

    for i, ch in enumerate(chunks, 1):
        meta = idx.doc_meta.get(ch["doc"], {})
        title = meta.get("title") or ch["doc"]
        try:
            items = json_array(await ask(_SYS, _prompt(ch["text"], title), max_tokens=900))
        except Exception as exc:  # noqa: BLE001 — log and continue
            print(f"  [{i}/{len(chunks)}] {ch['doc']} #{ch['ordinal']}: {exc}")
            continue

        cite = Citation(n=1, source=meta.get("source") or f"Local KB · {ch['doc']}",
                        url=meta.get("url"),
                        published=meta.get("revised") or meta.get("published"))
        prov = Provenance(generator="qa_from_corpus", model=MODEL, created_at=now,
                          corpus_doc=ch["doc"])

        made = 0
        for it in items:
            q, a = (it.get("question") or "").strip(), (it.get("answer") or "").strip()
            if len(q) < 8 or len(a) < 12:
                continue
            base = _cid(ch["doc"], str(ch["ordinal"]), q)
            qa_records.append(GuidelineQA(
                id=f"qa-{base}", question=q, answer=a, citations=[cite],
                recommendation_class=it.get("recommendation_class"),
                evidence_level=it.get("evidence_level"),
                provenance=prov,
            ))
            if sft_out is not None:
                sft_records.append(SFTExample(
                    id=f"sft-{base}", system=SYSTEM_PROMPT, user=q,
                    assistant=f"{a} [1]",
                    citations=[cite], tags=["guideline-qa", meta.get("category", "kb")],
                    provenance=prov,
                ))
            made += 1
        print(f"  [{i}/{len(chunks)}] {ch['doc']} #{ch['ordinal']} -> {made} Q&A")

    n = write_jsonl(out, qa_records)
    print(f"\nwrote {n} GuidelineQA -> {out}")
    if sft_out is not None:
        m = write_jsonl(sft_out, sft_records)
        print(f"wrote {m} SFTExample  -> {sft_out}")
    print("All records reviewed=false — run a clinician review before training.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=30, help="max passages to process (LLM calls)")
    ap.add_argument("--category", help="only passages under this knowledge/ subfolder")
    ap.add_argument("--out", default="data/guideline_qa.jsonl")
    ap.add_argument("--sft-out", default="data/sft_corpus.jsonl",
                    help="also write instruction examples here ('' to skip)")
    args = ap.parse_args(argv)
    asyncio.run(build(args.limit, args.category, args.out, args.sft_out or None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
