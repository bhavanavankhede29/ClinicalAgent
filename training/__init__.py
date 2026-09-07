"""Training-data pipeline for a case -> plan model.

This package does NOT train anything and ships NO patient data. It builds,
validates, and exports training examples:

    schemas.py        typed records — SFT, DPO preference, guideline Q&A, vignette
    qa_from_corpus.py  guideline/label chunks -> cited Q&A pairs (LLM, review-gated)
    vignettes.py       synthetic case -> graded plan + rationale (via the agent)
    preferences.py     A/B plans from two engine runs -> clinician-ranked pairs
    ehr_adapter.py     MIMIC-IV / eICU style stay -> CaseRecord  (you supply the path)
    export.py          collected records -> JSONL (sft | dpo | messages)

Every generated example carries `reviewed: false` and must be checked by a
clinician before it enters a training set. See README.md for governance.
"""
