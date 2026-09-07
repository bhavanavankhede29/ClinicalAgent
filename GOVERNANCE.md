# Governance

Controls that apply to **every** path — RAG corpus, fine-tuning data, and the
running service — before anything is patient-facing. Each has a code hook and an
artifact; process owners must sign off outside this repo.

| # | Control | Code hook | Artifact | Owner |
|---|---|---|---|---|
| 1 | **De-identification** (HIPAA Safe Harbor or Expert Determination) | `training/deident.py` — `scrub_text()`, `verify_case()`, `assert_case_clean()` | `governance/templates/deid_attestation_template.md` (signed, per source) | Privacy office |
| 2 | **IRB / DUA / licences** | `training/governance.py` — validates every source is registered, in scope, and not expired/placeholder | `governance/data_sources.json` (the registry) | Research compliance |
| 3 | **Dataset datasheets** | `training/datasheet.py` — generates a Gebru-style datasheet; `training/governance.py` fails release if one is missing | `governance/datasheets/<name>.md` | Data steward |
| 4 | **Documented review process** | `training/review.py` gates on `reviewed=true`; `training/export.py` excludes unreviewed and runs the governance gate by default | `governance/templates/release_checklist.md` (completed per release) | Clinical lead |
| 5 | **Drift monitoring** | `monitoring/drift.py` — `log_query()` (wired into `POST /api/query`) + `report` (PSI on input/output distributions) | `monitoring/requests.jsonl` (privacy-safe log) + periodic drift report | ML ops |
| 6 | **Model card** | — | `governance/templates/model_card_template.md` | Clinical lead + ML ops |

## Release gate (must all pass)

```powershell
# 1. sources registered, in scope, not expired
python -m training.governance check build/sft.jsonl --use training

# 2. datasheet exists for the export
python -m training.datasheet build/sft.jsonl --name sft --purpose "..." --dua "..." --irb "..." --deid "..."

# 3. every record clinician-reviewed
python -m training.review build/sft.jsonl --stats     # pending must be 0

# 4. export runs the gate itself (refuses on any failure)
python -m training.export build/sft.jsonl --format messages --out build/final_sft.jsonl

# 5. drift baseline established / green
python -m monitoring.drift report --baseline 30 --window 7
```

`--force-ungoverned` on `training.export` bypasses the gate for **dev builds only**
and must never be used for a set that will train a released model.

## De-identification standard

- Source data must be de-identified **before** it reaches this repo, by Safe
  Harbor (removal of all 18 identifier classes by a qualified person) or Expert
  Determination (documented statistical assessment). Record the method, the
  determiner, and the date in `governance/data_sources.json` and a signed
  attestation.
- `training/deident.py` is a **secondary** record-level check (dates, ages ≥ 90,
  SSN/MRN/phone/email/URL/IP in free text). It does not replace step 1.
- `CaseRecord` (schema) has no birth date, no admission dates, no notes — only an
  age band. `to_case_record()` enforces it.

## Registry

`governance/data_sources.json` lists every source with: access class, whether it
carries PHI, de-id method, agreement reference, IRB protocol, expiry, allowed
uses, custodian. Add a row **before** using a source; `FILL-IN` / `PER-PUBLISHER`
placeholders cause the governance gate to fail.
