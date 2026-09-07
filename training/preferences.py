"""Build A/B plan pairs for preference training (DPO / reward model).

For each scenario it runs two engines (or the same engine twice) to get two
candidate plans, then writes a PreferencePair with `chosen` provisionally set to
the first candidate and `reviewed=false`. A clinician then confirms or swaps with
`training.review`.

    python -m training.preferences --limit 6 --engines builtin,langchain \
        --out data/preferences.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import datetime, timezone

from api.models import QueryRequest

from .schemas import PreferencePair, Provenance, write_jsonl
from ._llm import MODEL, available
from .vignettes import DEFAULT_SEEDS


async def _run(engine: str, seed: str) -> str:
    req = QueryRequest(mode="decision_support", query=seed)
    if engine == "builtin":
        from api.agent import run_query
        return (await run_query(req)).answer_markdown.strip()
    if engine == "langchain":
        from api.lc_agent import LANGCHAIN_AVAILABLE, run_langchain_query
        if not LANGCHAIN_AVAILABLE:
            raise SystemExit("langchain engine not installed")
        return (await run_langchain_query(req)).answer_markdown.strip()
    if engine == "rag-graph":
        from api.rag_graph import RAG_GRAPH_AVAILABLE, run_rag_graph
        if not RAG_GRAPH_AVAILABLE:
            raise SystemExit("rag-graph engine not installed")
        return (await run_rag_graph(req)).answer_markdown.strip()
    raise SystemExit(f"unknown engine: {engine}")


async def build(seeds: list[str], engines: tuple[str, str], out: str) -> None:
    if not available():
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pairs: list[PreferencePair] = []
    for i, seed in enumerate(seeds, 1):
        a = await _run(engines[0], seed)
        b = await _run(engines[1], seed)
        if a == b:
            print(f"  [{i}] identical output, skipped: {seed[:50]}")
            continue
        pid = hashlib.sha1(seed.encode()).hexdigest()[:16]
        pairs.append(PreferencePair(
            id=f"pref-{pid}", prompt=seed, chosen=a, rejected=b,
            rationale=None, reviewed=False,
            provenance=Provenance(generator=f"preferences[{engines[0]} vs {engines[1]}]",
                                  model=MODEL, created_at=now, case_id=pid),
        ))
        print(f"  [{i}/{len(seeds)}] {seed[:55]}… -> pair")
    print(f"\nwrote {write_jsonl(out, pairs)} PreferencePair -> {out}")
    print("chosen is provisional — confirm/swap with `python -m training.review`.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-file")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--engines", default="builtin,langchain",
                    help="two of: builtin, langchain, rag-graph")
    ap.add_argument("--out", default="data/preferences.jsonl")
    args = ap.parse_args(argv)

    seeds = ([ln.strip() for ln in open(args.seeds_file, encoding="utf-8") if ln.strip()]
             if args.seeds_file else DEFAULT_SEEDS)
    e = tuple(x.strip() for x in args.engines.split(","))
    if len(e) != 2:
        raise SystemExit("--engines needs exactly two, comma-separated")
    asyncio.run(build(seeds[: args.limit], e, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
