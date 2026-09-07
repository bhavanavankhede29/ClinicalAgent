"""Request/response schemas. The response is deliberately verbose about
provenance: every claim the UI shows can be traced back to a retrieval step
and a cited source."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal[
    "comprehensive", "surveillance", "narratives", "treatment_research", "decision_support",
    "treatment_plan",
]

MODE_LABEL: dict[str, str] = {
    "comprehensive": "Comprehensive review",
    "surveillance": "Disease surveillance",
    "narratives": "Clinical narrative",
    "treatment_research": "Treatment research",
    "decision_support": "Clinical decision support",
    "treatment_plan": "Personalized treatment plan",
}

# The four sub-workflows a "comprehensive" answer is organised into, in order.
COMPREHENSIVE_SECTIONS: list[tuple[str, str]] = [
    ("decision_support", "Clinical decision support"),
    ("treatment_research", "Treatment research"),
    ("surveillance", "Disease surveillance"),
    ("narratives", "Clinical narrative"),
]


class QueryRequest(BaseModel):
    mode: Mode
    query: str = Field(min_length=3, max_length=2000)
    case_context: Optional[str] = Field(default=None, max_length=4000)
    # Optional: pull a synthetic patient's problem list / meds / results from the
    # sandbox FHIR server and use it as case context.
    fhir_patient_id: Optional[str] = Field(default=None, max_length=128)


class Citation(BaseModel):
    id: int
    source: str
    source_kind: str  # fda-label | literature | pharmacovigilance | nomenclature | local-knowledge
    title: str
    url: Optional[str] = None
    published: Optional[str] = None
    retrieved_at: str
    snippet: str
    # Present only for time-series evidence (surveillance signals).
    series: Optional[list[dict]] = None
    # Retrieval similarity score and method — set for local knowledge-base hits.
    score: Optional[float] = None
    retrieval: Optional[str] = None


class TraceStep(BaseModel):
    step: int
    tool: str
    source_api: str
    params: dict
    result_count: int
    latency_ms: int
    status: str  # ok | error
    note: Optional[str] = None


class LLMInfo(BaseModel):
    used: bool
    model: Optional[str] = None
    tool_rounds: int = 0
    reason: Optional[str] = None


class QueryResponse(BaseModel):
    mode: Mode
    query: str
    answer_markdown: str
    citations: list[Citation]
    retrieval_trace: list[TraceStep]
    llm: LLMInfo
    human_review_required: bool = True
    generated_at: str


class ThoughtItem(BaseModel):
    step: int
    agent: str          # Planner | Researcher | Synthesist | Safety Reviewer
    kind: str           # plan | thought | action | observation | draft | critique | decision
    content: str = ""
    data: Optional[dict] = None


class ThinkResponse(BaseModel):
    mode: Mode
    query: str
    thoughts: list[ThoughtItem]
    answer_markdown: str
    citations: list[Citation]
    retrieval_trace: list[TraceStep]
    llm: LLMInfo
    revisions: int = 0
    human_review_required: bool = True
    generated_at: str
