"""Synthetic case -> graded management plan vignettes.

For each scenario seed the pipeline:
  1. runs the evidence agent (`api.agent.run_query`) to get a cited answer,
  2. asks the LLM to render a case *stem* and *question* that answer addresses,
  3. stores a Vignette (reviewed=false) — draft material a clinician must vet.

    python -m training.vignettes --limit 8 --out data/vignettes.jsonl
    python -m training.vignettes --seeds-file seeds.txt --sft-out data/sft_vignettes.jsonl
    python -m training.vignettes --from-fhir 5        # use sandbox patients as stems
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import datetime, timezone

from api.agent import run_query
from api.models import QueryRequest

from .schemas import Citation, Provenance, SFTExample, SYSTEM_PROMPT, Vignette, write_jsonl
from ._llm import MODEL, ask, available, json_array

DEFAULT_SEEDS = [
    "new atrial fibrillation, CHA2DS2-VASc 4, eGFR 40 — anticoagulation choice",
    "type 2 diabetes, A1c 8.9%, established ASCVD, on metformin — add-on therapy",
    "community-acquired pneumonia, outpatient, penicillin allergy — antibiotic choice",
    "heart failure with reduced ejection fraction, NYHA II, on ACEi + beta-blocker — next agent",
    "acute uncomplicated cystitis in a non-pregnant adult — first-line antibiotic and duration",
    "COPD, two exacerbations this year despite LABA/LAMA — escalation options",
    "hypertension, stage 2, no compelling indication — initial regimen",
    "VTE (unprovoked proximal DVT), normal renal function — anticoagulant and duration",
]

_STEM_SYS = (
    "You write concise clinical case stems for training. Given a scenario and the "
    "evidence answer it should elicit, produce a JSON array with ONE object: "
    '{"stem": "<2-4 sentence de-identified case; use an age band not a birth date>", '
    '"question": "<the management question the answer addresses>", '
    '"difficulty": "basic|intermediate|advanced"}'
)


def _cid(seed: str) -> str:
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


async def _fhir_seeds(n: int) -> list[str]:
    from api.fhir import patient_context, search_patients
    seeds: list[str] = []
    for name in ("smith", "james", "rashmi", "anderson"):
        for p in await search_patients(name, count=8):
            if len(seeds) >= n:
                return seeds
            try:
                ctx = await patient_context(p["id"])
            except Exception:  # noqa: BLE001
                continue
            probs = "; ".join(x["name"] for x in ctx.get("conditions", [])[:4])
            meds = "; ".join(x["name"] for x in ctx.get("medications", [])[:4])
            if probs:
                seeds.append(f"{ctx['patient'].get('gender','adult')} with {probs}"
                             + (f", on {meds}" if meds else "")
                             + " — review current management and options")
    return seeds


async def build(seeds: list[str], out: str, sft_out: str | None) -> None:
    if not available():
        raise SystemExit("ANTHROPIC_API_KEY not set — cannot generate vignettes.")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    vignettes: list[Vignette] = []
    sft: list[SFTExample] = []

    for i, seed in enumerate(seeds, 1):
        resp = await run_query(QueryRequest(mode="decision_support", query=seed))
        plan = resp.answer_markdown.strip()
        cites = [Citation(n=c.id, source=c.source, url=c.url, published=c.published)
                 for c in resp.citations]

        meta = json_array(await ask(_STEM_SYS,
                                    f"Scenario: {seed}\n\nEvidence answer:\n{plan[:1500]}",
                                    max_tokens=500))
        m = meta[0] if meta else {}
        stem = (m.get("stem") or seed).strip()
        question = (m.get("question") or f"What is the recommended approach for: {seed}?").strip()

        base = _cid(seed)
        prov = Provenance(generator="vignettes", model=MODEL, created_at=now, case_id=base)
        vignettes.append(Vignette(
            id=f"vig-{base}", stem=stem, question=question, plan=plan,
            citations=cites, difficulty=m.get("difficulty"), provenance=prov,
        ))
        if sft_out is not None:
            sft.append(SFTExample(
                id=f"sft-vig-{base}", system=SYSTEM_PROMPT,
                user=f"{stem}\n\n{question}", assistant=plan,
                citations=cites, tags=["vignette", "case-plan"], provenance=prov,
            ))
        print(f"  [{i}/{len(seeds)}] {seed[:60]}… -> {len(cites)} citations")

    print(f"\nwrote {write_jsonl(out, vignettes)} Vignette -> {out}")
    if sft_out is not None:
        print(f"wrote {write_jsonl(sft_out, sft)} SFTExample -> {sft_out}")
    print("All records reviewed=false — clinician review required before training.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-file", help="one scenario per line (overrides defaults)")
    ap.add_argument("--from-fhir", type=int, default=0, help="use N sandbox patients as stems")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default="data/vignettes.jsonl")
    ap.add_argument("--sft-out", default="data/sft_vignettes.jsonl")
    args = ap.parse_args(argv)

    if args.seeds_file:
        seeds = [ln.strip() for ln in open(args.seeds_file, encoding="utf-8") if ln.strip()]
    elif args.from_fhir:
        seeds = asyncio.run(_fhir_seeds(args.from_fhir))
    else:
        seeds = DEFAULT_SEEDS
    asyncio.run(build(seeds[: args.limit], args.out, args.sft_out or None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
