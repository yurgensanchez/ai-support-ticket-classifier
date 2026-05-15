from pathlib import Path

import pandas as pd
import pytest

from ticket_classifier.ml.preprocessing import normalize_text
from ticket_classifier.ml.train import build_pipeline, format_metrics_summary, load_dataset, train_model


def test_normalize_text_strips_lowercases_and_collapses_spaces() -> None:
    assert normalize_text("  HELLO     Support\nTeam  ") == "hello support team"


def test_load_dataset_requires_text_and_category_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    dataset_path.write_text("message,label\nhello,other\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(dataset_path)


def test_build_pipeline_contains_expected_steps() -> None:
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]


def test_train_model_returns_metrics() -> None:
    dataset = pd.DataFrame(
        {
            "text": [
                "price information",
                "buy subscription",
                "discount question",
                "login error",
                "app crash",
                "dashboard broken",
                "invoice copy",
                "payment failed",
                "billing email",
                "bad service",
                "complaint delay",
                "unhappy customer",
                "book appointment",
                "schedule meeting",
                "change appointment",
                "general question",
                "office location",
                "opening hours",
            ],
            "category": [
                "sales",
                "sales",
                "sales",
                "technical_support",
                "technical_support",
                "technical_support",
                "billing",
                "billing",
                "billing",
                "complaint",
                "complaint",
                "complaint",
                "appointment",
                "appointment",
                "appointment",
                "other",
                "other",
                "other",
            ],
        }
    )

    model, metrics = train_model(dataset)

    assert hasattr(model, "predict")
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert sorted(metrics["labels"]) == [
        "appointment",
        "billing",
        "complaint",
        "other",
        "sales",
        "technical_support",
    ]


def test_format_metrics_summary_includes_confusion_matrix() -> None:
    metrics = {
        "accuracy": 0.5,
        "precision_weighted": 0.4,
        "recall_weighted": 0.5,
        "f1_weighted": 0.45,
        "train_rows": 20,
        "test_rows": 10,
        "labels": ["billing", "sales"],
        "confusion_matrix": [[2, 1], [1, 3]],
    }

    summary = format_metrics_summary(metrics)

    assert "# Model Evaluation Summary" in summary
    assert "| Actual \\ Predicted | billing | sales |" in summary
    assert "| billing | 2 | 1 |" in summary
