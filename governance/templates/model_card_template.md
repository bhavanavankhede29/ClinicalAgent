# Model card — <model name / version>

## Overview
- **Purpose:** evidence-grounded decision support (case → management options). Not
  a diagnostic device; not autonomous prescribing.
- **Base model / method:** __________  (fine-tune / DPO / RAG-only)
- **Owners:** clinical lead __________, ML ops __________

## Intended use / out of scope
- **Intended:** licensed clinicians, options + tradeoffs + citations, human review required.
- **Out of scope:** patient-facing use, definitive diagnosis, patient-specific dosing
  directive, populations not represented in eval, non-US labelling contexts (if trained on US SPL).

## Training data
- Sources: __________ (link datasheets in `governance/datasheets/`)
- Records: SFT ___, DPO ___; all clinician-reviewed. PHI: none.
- Known gaps / imbalances: __________

## Evaluation
- Benchmarks (MedQA / PubMedQA / …): __________
- Held-out institutional cases vs adjudicated acceptable plans: __________
- Safety / red-team results: __________
- Bias audit (age / sex / race matched cases): __________
- **Prospective / outcome-based validation:** status __________

## Limitations & risks
- EHR-derived supervision reflects historical practice, not optimal care.
- LLM synthesis can misstate a source nuance; citation + review is the mitigation.
- Performance degrades outside the represented case mix.

## Monitoring & maintenance
- Production drift: `monitoring.drift` — thresholds __________, owner __________
- Retrain trigger: __________   Rollback plan: __________
- Review cadence: __________
