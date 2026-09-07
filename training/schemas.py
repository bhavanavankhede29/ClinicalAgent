"""Typed training-example records and JSONL I/O.

Formats (see export.py for how each maps to a trainer):
  * SFTExample       -> instruction tuning  (messages: system/user/assistant)
  * PreferencePair   -> DPO / reward model  (prompt + chosen + rejected)
  * GuidelineQA      -> a Q&A derived from one guideline/label passage
  * Vignette         -> a synthetic case with a graded management plan
  * CaseRecord       -> a de-identified EHR stay, source for the above
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for licensed clinicians. "
    "Give evidence-grounded options and tradeoffs with [n] citations, surface "
    "contraindications and boxed warnings, never issue a patient-specific "
    "directive or a definitive diagnosis, and note that clinician review is "
    "required before any patient-impacting action."
)


class Citation(BaseModel):
    n: int
    source: str
    url: Optional[str] = None
    published: Optional[str] = None


class Provenance(BaseModel):
    generator: str                         # module that produced the example
    model: Optional[str] = None            # LLM used, if any
    created_at: str
    corpus_doc: Optional[str] = None       # knowledge/ file it derives from
    case_id: Optional[str] = None          # EHR / vignette id
    dataset: Optional[str] = None          # e.g. "mimic-iv-2.2"


class SFTExample(BaseModel):
    """One supervised instruction example."""
    id: str
    system: str = SYSTEM_PROMPT
    user: str
    assistant: str
    citations: list[Citation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reviewed: bool = False
    reviewer: Optional[str] = None
    provenance: Provenance


class PreferencePair(BaseModel):
    """One (prompt, chosen, rejected) example for DPO / reward modelling."""
    id: str
    system: str = SYSTEM_PROMPT
    prompt: str
    chosen: str
    rejected: str
    rationale: Optional[str] = None        # why chosen beats rejected
    margin: Optional[Literal["slight", "clear", "strong"]] = None
    reviewed: bool = False
    reviewer: Optional[str] = None
    provenance: Provenance


class GuidelineQA(BaseModel):
    id: str
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    recommendation_class: Optional[str] = None     # e.g. "Class I", "strong"
    evidence_level: Optional[str] = None           # e.g. "A", "moderate"
    reviewed: bool = False
    provenance: Provenance


class Vignette(BaseModel):
    id: str
    stem: str                             # the case presentation
    question: str
    plan: str                            # graded management plan + rationale
    citations: list[Citation] = Field(default_factory=list)
    difficulty: Optional[Literal["basic", "intermediate", "advanced"]] = None
    reviewed: bool = False
    provenance: Provenance


class CaseRecord(BaseModel):
    """A de-identified EHR stay — the raw material for vignettes / preferences.
    Never serialised with direct identifiers; dates are shifted, ages capped."""
    case_id: str
    dataset: str
    age_group: Optional[str] = None       # e.g. "65-74"; never a birth date
    sex: Optional[str] = None
    problems: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    labs: dict[str, str] = Field(default_factory=dict)
    procedures: list[str] = Field(default_factory=list)
    orders: list[str] = Field(default_factory=list)         # what was actually done
    outcomes: dict[str, str] = Field(default_factory=dict)  # los, readmit_30d, mortality…


# --- JSONL helpers -----------------------------------------------------------

def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, model: type[BaseModel]) -> Iterator[BaseModel]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def append_jsonl(path: str | Path, rec: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(rec.model_dump_json(exclude_none=True) + "\n")
