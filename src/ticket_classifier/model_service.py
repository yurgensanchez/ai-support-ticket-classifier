from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ticket_classifier.ml.preprocessing import normalize_text

DEFAULT_MODEL_PATH = Path("models/ticket_classifier.joblib")


class TicketClassifier:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def labels(self) -> list[str]:
        classifier = self.model.named_steps["classifier"]
        return [str(label) for label in classifier.classes_]

    def predict(self, text: str) -> dict[str, Any]:
        normalized = normalize_text(text)
        probabilities = self.model.predict_proba([normalized])[0]
        best_index = int(np.argmax(probabilities))
        labels = self.labels

        return {
            "category": labels[best_index],
            "confidence": float(probabilities[best_index]),
            "probabilities": {label: float(probability) for label, probability in zip(labels, probabilities)},
        }


def load_classifier(model_path: Path = DEFAULT_MODEL_PATH) -> TicketClassifier:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train-ticket-classifier before starting the API."
        )

    model = joblib.load(model_path)
    return TicketClassifier(model)


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    return load_classifier()
