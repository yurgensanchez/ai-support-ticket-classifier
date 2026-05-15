from __future__ import annotations

from pathlib import Path

import pandas as pd

from ticket_classifier.model_service import TicketClassifier


def predict_dataframe(
    dataframe: pd.DataFrame,
    classifier: TicketClassifier,
    text_column: str = "text",
) -> pd.DataFrame:
    """Add prediction columns to a dataframe of support messages."""
    if text_column not in dataframe.columns:
        raise ValueError(f"Input CSV must contain a '{text_column}' column")

    output = dataframe.copy()
    predictions = [classifier.predict(str(value)) for value in output[text_column]]

    output["predicted_category"] = [prediction["category"] for prediction in predictions]
    output["confidence"] = [prediction["confidence"] for prediction in predictions]

    labels = sorted({label for prediction in predictions for label in prediction["probabilities"]})
    for label in labels:
        output[f"probability_{label}"] = [
            prediction["probabilities"].get(label, 0.0) for prediction in predictions
        ]

    return output


def predict_csv(
    input_path: Path,
    output_path: Path,
    classifier: TicketClassifier,
    text_column: str = "text",
) -> Path:
    """Read a CSV, add predictions, and write a new CSV."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    predicted = predict_dataframe(dataframe, classifier, text_column=text_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_path, index=False)
    return output_path
