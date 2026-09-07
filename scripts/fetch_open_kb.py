"""Populate knowledge/drug-references/ with public-domain FDA drug labels.

Pulls each drug's current label from openFDA (same SPL data DailyMed serves,
keyless, public domain), writes a Markdown file with the treatment-relevant
sections and a .meta.json provenance sidecar, then reindexes.

    python scripts/fetch_open_kb.py
    python scripts/fetch_open_kb.py --drugs metformin,apixaban,ceftriaxone
    python scripts/fetch_open_kb.py --drugs-file mylist.txt --no-reindex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.config import KNOWLEDGE_DIR          # noqa: E402
from api.rag import reindex                   # noqa: E402

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DAILYMED_INFO = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={}"

DEFAULT_DRUGS = [
    "metformin", "insulin glargine", "empagliflozin", "semaglutide",
    "lisinopril", "losartan", "amlodipine", "metoprolol succinate", "furosemide",
    "atorvastatin", "apixaban", "warfarin", "clopidogrel",
    "amoxicillin", "amoxicillin and clavulanate", "azithromycin", "doxycycline",
    "ceftriaxone", "levofloxacin", "vancomycin",
    "prednisone", "albuterol", "tiotropium", "sertraline", "gabapentin",
    "levothyroxine", "omeprazole", "ondansetron", "acetaminophen", "ibuprofen",
]

# openFDA label field -> Markdown heading, in the order we want them
SECTIONS = [
    ("boxed_warning", "Boxed warning"),
    ("indications_and_usage", "Indications and usage"),
    ("dosage_and_administration", "Dosage and administration"),
    ("dosage_forms_and_strengths", "Dosage forms and strengths"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and precautions"),
    ("warnings", "Warnings"),
    ("drug_interactions", "Drug interactions"),
    ("use_in_specific_populations", "Use in specific populations"),
    ("pregnancy", "Pregnancy"),
    ("pediatric_use", "Pediatric use"),
    ("geriatric_use", "Geriatric use"),
    ("renal_impairment", "Renal impairment"),
    ("hepatic_impairment", "Hepatic impairment"),
    ("overdosage", "Overdosage"),
]
_SECTION_CAP = 8000


def _text(value) -> str:
    if not value:
        return ""
    joined = "\n\n".join(value) if isinstance(value, list) else str(value)
    joined = re.sub(r"[ \t]+", " ", joined).strip()
    return joined[:_SECTION_CAP].rstrip() + ("…" if len(joined) > _SECTION_CAP else "")


def _first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value or None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "drug"


def _fmt_date(spl_date: str | None) -> str | None:
    if spl_date and re.fullmatch(r"\d{8}", spl_date):
        return f"{spl_date[:4]}-{spl_date[4:6]}-{spl_date[6:]}"
    return spl_date or None


def fetch_label(client: httpx.Client, drug: str) -> dict | None:
    dl = drug.lower()

    first_word = dl.split()[0]

    def rank(res: dict) -> tuple:
        g = [x.lower() for x in res.get("openfda", {}).get("generic_name", [])]
        joined = " ; ".join(g)
        mono = not any(sep in joined for sep in (" and ", " with ", "/"))  # not a combo product
        starts = any(x.startswith(first_word) for x in g)
        match = any(dl in x or x in dl for x in g)
        return (mono, starts, match, len(g) == 1)

    for field in ("generic_name", "brand_name"):
        r = client.get(FDA_LABEL_URL,
                       params={"search": f'openfda.{field}:"{drug}"', "limit": 8})
        if r.status_code != 200:
            continue
        results = r.json().get("results", [])
        if results:
            results.sort(key=rank, reverse=True)
            return results[0]
    return None


def build_markdown(drug: str, label: dict) -> tuple[str, dict]:
    fda = label.get("openfda", {})
    generic = _first(fda.get("generic_name")) or drug
    brand = _first(fda.get("brand_name"))
    manuf = _first(fda.get("manufacturer_name"))
    set_id = _first(fda.get("spl_set_id"))
    effective = _fmt_date(_first(label.get("effective_time")))
    title = f"{generic.title()}" + (f" ({brand})" if brand and brand.lower() != generic.lower() else "")

    lines = [f"# {title} — FDA Prescribing Information", ""]
    if manuf:
        lines += [f"Labeler: {manuf}.", ""]
    seen = set()
    for field, heading in SECTIONS:
        body = _text(label.get(field))
        if not body or heading in seen:
            continue
        seen.add(heading)
        lines += [f"## {heading}", "", body, ""]

    meta = {
        "source": "FDA drug label (openFDA / DailyMed)",
        "title": f"{title} — FDA Prescribing Information",
        "category": "drug-references",
        "published": effective,
        "version": f"SPL effective {effective}" if effective else None,
        "jurisdiction": "US",
        "url": DAILYMED_INFO.format(set_id) if set_id else "https://dailymed.nlm.nih.gov/",
    }
    return "\n".join(lines), {k: v for k, v in meta.items() if v}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", help="comma-separated drug names (overrides the default set)")
    ap.add_argument("--drugs-file", help="file with one drug name per line")
    ap.add_argument("--no-reindex", action="store_true")
    args = ap.parse_args(argv)

    if args.drugs:
        drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    elif args.drugs_file:
        drugs = [ln.strip() for ln in Path(args.drugs_file).read_text("utf-8").splitlines() if ln.strip()]
    else:
        drugs = DEFAULT_DRUGS

    dest = KNOWLEDGE_DIR / "drug-references"
    dest.mkdir(parents=True, exist_ok=True)
    ok, miss = [], []
    with httpx.Client(timeout=30.0, headers={"User-Agent": "clinical-agent-kb/1.0"}) as client:
        for drug in drugs:
            try:
                label = fetch_label(client, drug)
            except httpx.HTTPError as exc:
                print(f"  ! {drug}: {exc}")
                miss.append(drug)
                continue
            if not label:
                print(f"  - {drug}: no FDA label found")
                miss.append(drug)
                continue
            md, meta = build_markdown(drug, label)
            stem = _slug(drug)
            (dest / f"{stem}.md").write_text(md, encoding="utf-8")
            (dest / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  + {drug}  ->  drug-references/{stem}.md  ({meta.get('published', 'n.d.')})")
            ok.append(drug)
            time.sleep(0.3)  # be polite to openFDA

    print(f"\n{len(ok)} labels written, {len(miss)} missed"
          + (f": {', '.join(miss)}" if miss else ""))
    if not args.no_reindex and ok:
        stats = reindex()
        print("reindexed:", json.dumps({k: stats.get(k) for k in ("documents", "chunks", "mode", "model")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
