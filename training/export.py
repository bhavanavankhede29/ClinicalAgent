"""Export reviewed training records to a trainer-ready JSONL format.

    python -m training.export data/sft_corpus.jsonl data/sft_vignettes.jsonl \
        --format messages --out build/sft.jsonl
    python -m training.export data/preferences.jsonl --format dpo --out build/dpo.jsonl

Formats:
  messages  {"messages": [ {role, content} x3 ]}            (SFT: OpenAI / Anthropic style)
  sft       {"system","prompt","completion"}                (flat SFT)
  dpo       {"system","prompt","chosen","rejected"}         (DPO / preference)

Only records with reviewed=true are exported unless --include-unreviewed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    return out


def _to_messages(r: dict) -> dict | None:
    if "user" in r and "assistant" in r:                       # SFTExample
        return {"messages": [
            {"role": "system", "content": r.get("system", "")},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["assistant"]},
        ]}
    if {"question", "answer"} <= r.keys():                      # GuidelineQA
        return {"messages": [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ]}
    if {"stem", "question", "plan"} <= r.keys():                # Vignette
        return {"messages": [
            {"role": "user", "content": f"{r['stem']}\n\n{r['question']}"},
            {"role": "assistant", "content": r["plan"]},
        ]}
    return None


def _to_sft(r: dict) -> dict | None:
    if "user" in r and "assistant" in r:
        return {"system": r.get("system", ""), "prompt": r["user"], "completion": r["assistant"]}
    if {"question", "answer"} <= r.keys():
        return {"system": "", "prompt": r["question"], "completion": r["answer"]}
    if {"stem", "question", "plan"} <= r.keys():
        return {"system": "", "prompt": f"{r['stem']}\n\n{r['question']}", "completion": r["plan"]}
    return None


def _to_dpo(r: dict) -> dict | None:
    if {"prompt", "chosen", "rejected"} <= r.keys():
        return {"system": r.get("system", ""), "prompt": r["prompt"],
                "chosen": r["chosen"], "rejected": r["rejected"]}
    return None


CONVERTERS = {"messages": _to_messages, "sft": _to_sft, "dpo": _to_dpo}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--format", choices=list(CONVERTERS), default="messages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--use", default="training", help="governance scope: training | eval")
    ap.add_argument("--force-ungoverned", action="store_true",
                    help="DEV ONLY — skip the governance gate")
    args = ap.parse_args(argv)

    if not args.force_ungoverned:
        from .governance import check_release
        ok, issues = check_release(list(args.inputs), args.use)
        if not ok:
            print("Refusing to export — governance gate failed:")
            for i in issues:
                print(f"  - {i}")
            print("Fix the above, or pass --force-ungoverned for a dev build.")
            return 1

    conv = CONVERTERS[args.format]
    rows = _rows(args.inputs)
    kept, skipped_unreviewed, skipped_shape = [], 0, 0
    for r in rows:
        if not args.include_unreviewed and not r.get("reviewed"):
            skipped_unreviewed += 1
            continue
        rec = conv(r)
        if rec is None:
            skipped_shape += 1
            continue
        kept.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")
    print(f"wrote {len(kept)} -> {out}  (format={args.format})")
    if skipped_unreviewed:
        print(f"  skipped {skipped_unreviewed} unreviewed (use --include-unreviewed to force)")
    if skipped_shape:
        print(f"  skipped {skipped_shape} not convertible to {args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
