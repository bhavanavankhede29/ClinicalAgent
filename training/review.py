"""Mark training records as clinician-reviewed, or fix preference pairs.

    python -m training.review data/sft_corpus.jsonl --stats
    python -m training.review data/sft_corpus.jsonl --approve ID1,ID2 --reviewer "Dr X"
    python -m training.review data/sft_corpus.jsonl --approve-all --reviewer "Dr X"   # demo only
    python -m training.review data/preferences.jsonl --swap pref-abc123 --rationale "safer in CKD"
    python -m training.review data/preferences.jsonl --reject pref-def456   # drop from set

Approve only what a clinician has actually read. --approve-all exists for
pipeline demos, not for real datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--approve", help="comma-separated ids to mark reviewed")
    ap.add_argument("--approve-all", action="store_true")
    ap.add_argument("--reject", help="comma-separated ids to drop")
    ap.add_argument("--swap", help="preference id: swap chosen<->rejected")
    ap.add_argument("--rationale", help="set on the --swap'd (or --approve'd single) pair")
    ap.add_argument("--reviewer", default="unspecified")
    args = ap.parse_args(argv)

    path = Path(args.file)
    rows = _load(path)
    ids = lambda s: {x.strip() for x in (s or "").split(",") if x.strip()}
    approve, reject = ids(args.approve), ids(args.reject)
    changed = 0

    if args.stats:
        total = len(rows)
        done = sum(1 for r in rows if r.get("reviewed"))
        print(f"{path.name}: {total} records, {done} reviewed, {total - done} pending")
        return 0

    if reject:
        rows = [r for r in rows if r.get("id") not in reject]
        changed += 1

    for r in rows:
        rid = r.get("id")
        if args.swap and rid == args.swap and "chosen" in r:
            r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
            r["reviewed"] = True
            r["reviewer"] = args.reviewer
            if args.rationale:
                r["rationale"] = args.rationale
            changed += 1
        elif args.approve_all or rid in approve:
            r["reviewed"] = True
            r["reviewer"] = args.reviewer
            if args.rationale and (rid in approve):
                r["rationale"] = args.rationale
            changed += 1

    _save(path, rows)
    print(f"updated {changed} record(s) in {path.name}; "
          f"{sum(1 for r in rows if r.get('reviewed'))}/{len(rows)} now reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
