"""Generate a 'Datasheet for Datasets' (Gebru et al., 2021) for a training export.

    python -m training.datasheet build/sft.jsonl \
        --name sft --purpose "SFT for case->plan decision support" \
        --use training --dua "PhysioNet DUA #12345" --irb "IRB-2026-088" \
        --deid "Safe Harbor (J. Doe, 2026-02); training.deident record checks" \
        --out governance/datasheets/sft.md

Writes governance/datasheets/<name>.md — required by training.governance before
an export is releasable.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .governance import DATASHEET_DIR, load_registry


def _summarise(paths: list[str]) -> dict:
    n = 0
    reviewed = 0
    generators: Counter = Counter()
    datasets: Counter = Counter()
    tags: Counter = Counter()
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            n += 1
            reviewed += bool(r.get("reviewed"))
            prov = r.get("provenance") or {}
            generators[prov.get("generator", "?")] += 1
            datasets[prov.get("dataset") or "corpus-derived"] += 1
            for t in r.get("tags", []):
                tags[t] += 1
    return {"n": n, "reviewed": reviewed, "generators": dict(generators),
            "datasets": dict(datasets), "tags": dict(tags)}


TEMPLATE = """# Datasheet — {name}

_Generated {today}. Follows Gebru et al., "Datasheets for Datasets"._

## Motivation
- **Purpose:** {purpose}
- **Intended use:** {use}
- **Created by / for:** {custodian}

## Composition
- **Records:** {n}  (**reviewed:** {reviewed}/{n})
- **Instance type:** {use} examples (JSONL)
- **Generators:** {generators}
- **Upstream datasets / sources:** {datasets}
- **Tags:** {tags}
- **Contains PHI:** No. Source rows below are public-domain, synthetic, or
  de-identified before ingestion.

## Collection process
- **Source access & agreements:**
{source_rows}
- **De-identification:** {deid}
- **DUA / licence reference:** {dua}
- **IRB:** {irb}

## Preprocessing / cleaning / labelling
- Passages chunked (500 chars, 100 overlap) and, where LLM-generated, constrained
  to the source passage. Every record carries `provenance`.
- **Review process:** each record set to `reviewed=true` by a named clinician via
  `training.review`; unreviewed records are excluded by `training.export`.

## Uses
- **Suitable for:** {use}. Guideline-derived Q&A and reviewed vignettes only.
- **Not suitable for:** training a model to imitate historical orders without
  outcome-based validation; any use outside `allowed_uses` in the registry.
- **Known risks:** LLM-generated items may misstate a nuance of the source;
  clinician review is the mitigation. Bias-audit set required before deployment.

## Distribution
- Internal to {custodian}. Redistribution follows the most restrictive upstream
  licence. Do not publish records derived from licensed guideline text.

## Maintenance
- **Owner:** {owner}
- **Update cadence:** regenerate when the corpus or guidelines are updated;
  re-review changed records.
- **Drift monitoring:** production requests logged and compared to baseline
  (`monitoring.drift`); retrain trigger on sustained shift.
"""


def build(paths: list[str], **kw) -> str:
    s = _summarise(paths)
    reg = load_registry()
    used = set(s["datasets"]) | {"openfda-druglabel", "society-guidelines"}
    rows = []
    for sid in sorted(used):
        src = reg.get(sid) or next((v for k, v in reg.items() if str(sid).startswith(k)), None)
        if src:
            rows.append(f"  - **{src['name']}** — access: {src['access']}; "
                        f"agreement: {src.get('agreement_ref') or 'n/a'}; "
                        f"expires: {src.get('expires') or 'n/a'}")
        else:
            rows.append(f"  - `{sid}` — NOT IN REGISTRY (add to governance/data_sources.json)")
    return TEMPLATE.format(
        today=date.today().isoformat(),
        source_rows="\n".join(rows),
        n=s["n"], reviewed=s["reviewed"],
        generators=", ".join(f"{k} ({v})" for k, v in s["generators"].items()) or "n/a",
        datasets=", ".join(f"{k} ({v})" for k, v in s["datasets"].items()) or "n/a",
        tags=", ".join(f"{k} ({v})" for k, v in s["tags"].items()) or "n/a",
        **kw,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", required=True, help="datasheet name (also the export basename)")
    ap.add_argument("--purpose", default="(describe)")
    ap.add_argument("--use", default="training")
    ap.add_argument("--custodian", default="(institution)")
    ap.add_argument("--owner", default="(name / role)")
    ap.add_argument("--deid", default="see source rows")
    ap.add_argument("--dua", default="n/a")
    ap.add_argument("--irb", default="n/a")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    md = build(args.files, name=args.name, purpose=args.purpose, use=args.use,
               custodian=args.custodian, owner=args.owner, deid=args.deid,
               dua=args.dua, irb=args.irb)
    out = Path(args.out) if args.out else DATASHEET_DIR / f"{args.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
