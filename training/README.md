# Training-data pipeline

Builds, reviews, and exports training examples for a **case → plan** model. It
does **not** train a model and ships **no patient data**.

```
schemas.py         typed records + JSONL I/O
qa_from_corpus.py  guideline / drug-label passages -> cited Q&A + SFT examples
vignettes.py       synthetic case -> graded plan (runs the evidence agent)
preferences.py     two engine runs -> A/B pairs for DPO / reward modelling
review.py          mark records reviewed / swap or drop preference pairs
export.py          reviewed records -> messages | sft | dpo JSONL
ehr_adapter.py     de-identified EHR stay -> CaseRecord  (you supply the path)
```

## Pipeline

```powershell
# 1. instruction data from the RAG corpus (needs ANTHROPIC_API_KEY + an indexed corpus)
python -m training.qa_from_corpus --limit 40 --out data/qa.jsonl --sft-out data/sft_corpus.jsonl

# 2. synthetic case -> plan vignettes
python -m training.vignettes --limit 8 --out data/vignettes.jsonl --sft-out data/sft_vig.jsonl
python -m training.vignettes --from-fhir 5      # use sandbox patients as case stems

# 3. preference pairs (two engines disagree -> a ranking task)
python -m training.preferences --engines builtin,rag-graph --out data/prefs.jsonl

# 4. clinician review  (nothing is training-ready until this step)
python -m training.review data/sft_corpus.jsonl --stats
python -m training.review data/sft_corpus.jsonl --approve sft-abc,sft-def --reviewer "Dr X"
python -m training.review data/prefs.jsonl --swap pref-123 --rationale "safer with reduced eGFR"

# 5. export
python -m training.export data/sft_corpus.jsonl data/sft_vig.jsonl --format messages --out build/sft.jsonl
python -m training.export data/prefs.jsonl --format dpo --out build/dpo.jsonl
```

## Getting the datasets (your responsibility)

| Dataset | Access |
| --- | --- |
| **MIMIC-IV / MIMIC-IV-ED / eICU-CRD** | PhysioNet account + CITI "Data or Specimens Only Research" training + signed DUA. `physionet.org/content/mimiciv/` |
| **Institutional EHR** | IRB approval + de-identification (HIPAA Safe Harbor or Expert Determination) **before** the data reaches `ehr_adapter.py` |
| **Claims** (CMS LDS/ResDAC, MarketScan) | Their DUAs and fees |
| **SEER** | NCI SEER research data agreement |
| **USMLE/MRCP-style banks** | Licensed question banks — check redistribution terms before deriving examples |

Point an adapter at your local copy:
`python -m training.ehr_adapter --root D:\mimic-iv-2.2 --limit 3`

## Governance

- Every generated record is `reviewed: false`. A clinician approves each one
  (`training.review`) before it enters a training set. `--approve-all` is for
  pipeline demos only.
- `CaseRecord` carries an **age band**, not a birth date; **no admission dates**;
  **no free-text notes**. `to_case_record()` enforces this.
- Keep `provenance` on every record (generator, model, source doc / case id,
  dataset + version) — you need it for dataset datasheets and audits.
- Log a bias-audit set: matched cases varying age / sex / race, checked for
  disparate plans.

## The load-bearing caveat

EHR "treatment that was given" ≠ "treatment that was optimal". A model trained to
imitate historical orders learns clinician habits, local formulary quirks,
**confounding by indication**, and existing care disparities. Guideline-derived
Q&A and reviewed vignettes are safer supervision than raw order imitation, and
any trained component needs **prospective, outcome-based validation**, not just
accuracy against past decisions.
