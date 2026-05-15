from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from ticket_classifier.api.schemas import PredictionRequest, PredictionResponse
from ticket_classifier.model_service import TicketClassifier, get_classifier

app = FastAPI(
    title="AI Support Ticket Classifier",
    description="Classifies support messages into business-oriented categories.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(
    request: PredictionRequest,
    classifier: TicketClassifier = Depends(get_classifier),
) -> PredictionResponse:
    try:
        prediction = classifier.predict(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(**prediction)
