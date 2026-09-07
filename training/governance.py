"""Governance gate for the training pipeline.

Validates that a set of produced JSONL files is releasable:
  1. every `provenance.dataset` (and the generator's implied sources) is listed
     in governance/data_sources.json,
  2. that source is in scope for the use (training / eval), and its agreement
     has not expired or been left as a FILL-IN placeholder,
  3. every record is `reviewed: true`,
  4. a datasheet exists for the export (governance/datasheets/<name>.md).

    python -m training.governance check build/sft.jsonl build/dpo.jsonl
    python -m training.governance sources        # print the registry

`export.py` calls check_release() by default; use --force-ungoverned for dev only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = _ROOT / "governance" / "data_sources.json"
DATASHEET_DIR = _ROOT / "governance" / "datasheets"

# knowledge/ subfolder -> registry source id
_CORPUS_PREFIX_SOURCE = {
    "drug-references/": "openfda-druglabel",
    "guidelines/": "society-guidelines",
    "reviews/": "systematic-reviews",
    "protocols/": "institutional-docs",
    "local/": "institutional-docs",
}
# fallback when a record has no corpus_doc (e.g. a vignette built from a scenario seed)
_GENERATOR_SOURCES = {
    "qa_from_corpus": {"openfda-druglabel"},
    "vignettes": {"openfda-druglabel", "society-guidelines", "fhir-sandbox-hapi"},
    "preferences": {"openfda-druglabel", "society-guidelines"},
}
_PLACEHOLDER = ("FILL-IN", "PER-PUBLISHER")


def _source_for_doc(doc: str) -> str | None:
    for prefix, sid in _CORPUS_PREFIX_SOURCE.items():
        if doc.startswith(prefix):
            return sid
    return None


def load_registry() -> dict[str, dict]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["sources"]}


def _source_ok(src: dict, use: str) -> list[str]:
    problems = []
    if use not in src.get("allowed_uses", []):
        problems.append(f"{src['id']}: use '{use}' not in allowed_uses {src.get('allowed_uses')}")
    for field in ("agreement_ref", "expires", "deid_method", "irb_protocol"):
        val = src.get(field)
        if isinstance(val, str) and any(p in val for p in _PLACEHOLDER):
            problems.append(f"{src['id']}: {field} is a placeholder — fill it in")
    exp = src.get("expires")
    if isinstance(exp, str) and not any(p in exp for p in _PLACEHOLDER):
        try:
            if date.fromisoformat(exp[:10]) < date.today():
                problems.append(f"{src['id']}: agreement expired {exp}")
        except ValueError:
            problems.append(f"{src['id']}: expires '{exp}' is not an ISO date")
    return problems


def _records(paths: list[str]):
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                yield p, json.loads(ln)


def check_release(paths: list[str], use: str = "training") -> tuple[bool, list[str]]:
    reg = load_registry()
    issues: list[str] = []
    used_sources: set[str] = set()
    total = reviewed = 0

    for p, rec in _records(paths):
        total += 1
        if rec.get("reviewed"):
            reviewed += 1
        prov = rec.get("provenance") or {}
        if prov.get("dataset"):
            used_sources.add(prov["dataset"])
        doc_src = _source_for_doc(prov.get("corpus_doc") or "")
        if doc_src:
            used_sources.add(doc_src)
        elif not prov.get("dataset"):
            gen = (prov.get("generator") or "").split("[")[0]
            used_sources |= _GENERATOR_SOURCES.get(gen, set())

    if reviewed < total:
        issues.append(f"{total - reviewed}/{total} records are not reviewed=true")

    for sid in sorted(used_sources):
        # tolerate a dataset id like 'mimic-iv-2.2' mapping to registry 'mimic-iv'
        src = reg.get(sid) or next((v for k, v in reg.items() if sid.startswith(k)), None)
        if src is None:
            issues.append(f"source '{sid}' is not in governance/data_sources.json")
        else:
            issues += _source_ok(src, use)

    for p in paths:
        name = Path(p).stem
        if not (DATASHEET_DIR / f"{name}.md").exists():
            issues.append(f"no datasheet: governance/datasheets/{name}.md "
                          f"(run `python -m training.datasheet {p} ...`)")

    return (not issues), issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    c.add_argument("--use", default="training", choices=["training", "eval", "rag-corpus", "demo"])
    sub.add_parser("sources")
    args = ap.parse_args(argv)

    if args.cmd == "sources":
        for sid, s in load_registry().items():
            print(f"- {sid:22} {s['access']:18} uses={s['allowed_uses']} expires={s.get('expires')}")
        return 0

    ok, issues = check_release(args.files, args.use)
    if ok:
        print("GOVERNANCE OK — releasable.")
        return 0
    print("GOVERNANCE FAILED:", file=sys.stderr)
    for i in issues:
        print(f"  - {i}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
