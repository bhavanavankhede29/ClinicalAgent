# De-identification attestation

**Source:** _(registry id from governance/data_sources.json, e.g. `institutional-ehr`)_
**Extract / version:** _____________________
**Date range of data:** _____________________  (dates shifted: yes / no)

## Method  (choose one)

- [ ] **Safe Harbor** — all 18 HIPAA identifier classes removed:
  names; geographic subdivisions smaller than state; all date elements (except
  year) directly related to an individual, and year for ages > 89; telephone /
  fax; email; SSN; MRN; health-plan beneficiary #; account #; certificate /
  licence #; vehicle identifiers; device identifiers; URLs; IP addresses;
  biometric identifiers; full-face photos; any other unique identifying number,
  characteristic, or code.
  Performed by: ________________  Role: ________________  Date: __________

- [ ] **Expert Determination** — a qualified statistician / scientist has
  determined the re-identification risk is very small.
  Expert: ________________  Method summary / report ref: ________________
  Date: __________

## Record-level verification

- [ ] `training.deident.verify_case()` run on a sample of ___ records — findings: ______
- [ ] `training.deident.scrub_text()` applied to any free-text fields retained

## Approvals

| Role | Name | Signature | Date |
|---|---|---|---|
| Data steward | | | |
| Privacy office | | | |
| Research compliance (IRB/DUA) | | | |
