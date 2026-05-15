from pathlib import Path

import pandas as pd
import pytest

from ticket_classifier.batch import predict_csv, predict_dataframe


class FakeClassifier:
    def predict(self, text: str) -> dict[str, object]:
        if "invoice" in text:
            return {
                "category": "billing",
                "confidence": 0.8,
                "probabilities": {"billing": 0.8, "sales": 0.2},
            }
        return {
            "category": "sales",
            "confidence": 0.7,
            "probabilities": {"billing": 0.3, "sales": 0.7},
        }


def test_predict_dataframe_adds_prediction_columns() -> None:
    dataframe = pd.DataFrame({"text": ["send invoice", "pricing info"]})

    result = predict_dataframe(dataframe, FakeClassifier())

    assert list(result["predicted_category"]) == ["billing", "sales"]
    assert list(result["confidence"]) == [0.8, 0.7]
    assert "probability_billing" in result.columns
    assert "probability_sales" in result.columns


def test_predict_dataframe_requires_text_column() -> None:
    dataframe = pd.DataFrame({"message": ["hello"]})

    with pytest.raises(ValueError, match="must contain a 'text' column"):
        predict_dataframe(dataframe, FakeClassifier())


def test_predict_csv_writes_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "tickets.csv"
    output_path = tmp_path / "predictions.csv"
    input_path.write_text("text\nsend invoice\npricing info\n", encoding="utf-8")

    result_path = predict_csv(input_path, output_path, FakeClassifier())

    assert result_path == output_path.resolve()
    output = pd.read_csv(output_path)
    assert list(output["predicted_category"]) == ["billing", "sales"]
