from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    score: float
    strategy: Optional[str] = None
    language: Optional[str] = None
    quote: str


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    strategy: Optional[str] = None
    language: Optional[str] = None
    parent_doc_id: Optional[str] = None
    title: Optional[str] = None


class RagResponse(BaseModel):
    transcript: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    abstained: bool = False
    abstain_reason: Optional[str] = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class TextRequest(BaseModel):
    query: str
