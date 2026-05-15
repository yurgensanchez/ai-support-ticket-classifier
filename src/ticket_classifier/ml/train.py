from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ticket_classifier.ml.preprocessing import normalize_text

DEFAULT_DATASET_PATH = Path("data/sample_tickets.csv")
DEFAULT_MODEL_PATH = Path("models/ticket_classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_SUMMARY_PATH = Path("models/metrics_summary.md")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load and validate the ticket dataset."""
    dataset = pd.read_csv(dataset_path)
    required_columns = {"text", "category"}
    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataset = dataset.dropna(subset=["text", "category"]).copy()
    dataset["text"] = dataset["text"].map(normalize_text)
    dataset["category"] = dataset["category"].astype(str).str.strip()

    if dataset.empty:
        raise ValueError("Dataset does not contain valid training rows")

    return dataset


def build_pipeline() -> Pipeline:
    """Build the baseline text classification pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(dataset: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """Train a model and return the fitted pipeline with evaluation metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["category"],
        test_size=0.3,
        random_state=42,
        stratify=dataset["category"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(dataset["category"].unique().tolist())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "confusion_matrix": matrix.tolist(),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "labels": labels,
    }

    return model, metrics


def save_model(model: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(metrics: dict[str, Any], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Create a human-readable Markdown summary for training results."""
    lines = [
        "# Model Evaluation Summary",
        "",
        "This summary is generated from the local sample dataset. The dataset is small, so these metrics should be read as a demo signal rather than real production evidence.",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Weighted precision: {metrics['precision_weighted']:.3f}",
        f"- Weighted recall: {metrics['recall_weighted']:.3f}",
        f"- Weighted F1: {metrics['f1_weighted']:.3f}",
        f"- Train rows: {metrics['train_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        "",
        "## Confusion Matrix",
        "",
    ]

    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    lines.append("| Actual \\ Predicted | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(labels)) + " |")

    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Low scores are expected with this tiny fictional dataset.",
            "- More labeled examples are needed before the model can be trusted for real support routing.",
            "- The goal of this project is to demonstrate a complete baseline ML workflow.",
        ]
    )

    return "\n".join(lines) + "\n"


def save_metrics_summary(metrics: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(format_metrics_summary(metrics), encoding="utf-8")


def run_training(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    model, metrics = train_model(dataset)
    save_model(model, model_path)
    save_metrics(metrics, metrics_path)
    save_metrics_summary(metrics, summary_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the support ticket classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH, help="Path to CSV training dataset.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path where the model will be saved.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Path where metrics will be saved.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="Path where Markdown metrics summary will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(args.data, args.model, args.metrics, args.summary)
    print(f"Model saved to: {args.model}")
    print(f"Metrics saved to: {args.metrics}")
    print(f"Summary saved to: {args.summary}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()
