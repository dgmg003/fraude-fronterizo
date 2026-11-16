from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..domain.transaction import Transaction

DISCLAIMER = (
    "Las alertas son apoyo para la revisión humana y no constituyen "
    "un dictamen de fraude."
)


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=10_000)


class ScoredTransaction(BaseModel):
    transaction_id: str
    anomaly_score: float
    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["bajo", "medio", "alto"]
    requires_review: bool


class ScoreResponse(BaseModel):
    disclaimer: str = DISCLAIMER
    results: list[ScoredTransaction]
