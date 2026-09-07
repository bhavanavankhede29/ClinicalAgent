# Release checklist — training set / model update

**Release id:** ____________   **Date:** __________   **Prepared by:** ____________

## Data
- [ ] Every source used is a row in `governance/data_sources.json`
- [ ] No `FILL-IN` / `PER-PUBLISHER` placeholders on used sources
- [ ] All DUAs / licences current (expiry >= today)
- [ ] `python -m training.governance check <exports> --use training` → **OK**
- [ ] De-identification attestation on file for each restricted source
- [ ] Datasheet generated and reviewed: `governance/datasheets/<name>.md`

## Review
- [ ] `python -m training.review <export> --stats` → **0 pending**
- [ ] Reviewer names recorded on records (not "unspecified")
- [ ] Bias-audit set run — matched cases varying age / sex / race; no disparate plans
- [ ] Safety / red-team set run — contraindication traps, allergy conflicts, dose errors

## Model
- [ ] Model card completed (`governance/templates/model_card_template.md`)
- [ ] Held-out clinical eval passed threshold: __________
- [ ] Prospective / outcome-based validation plan documented (not just past-decision accuracy)

## Monitoring
- [ ] `monitoring/requests.jsonl` logging confirmed in the target environment
- [ ] Drift baseline captured: `python -m monitoring.drift report` → green
- [ ] Retrain / rollback trigger and owner defined

## Sign-off
| Role | Name | Date |
|---|---|---|
| Clinical lead | | |
| ML ops | | |
| Privacy / compliance | | |
