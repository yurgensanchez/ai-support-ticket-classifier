from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000, description="Support ticket message to classify.")


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]
