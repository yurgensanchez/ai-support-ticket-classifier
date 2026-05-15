from __future__ import annotations

import argparse
from pathlib import Path

from ticket_classifier.batch import predict_csv
from ticket_classifier.model_service import DEFAULT_MODEL_PATH, load_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions for support ticket messages.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with a text column.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path for predictions.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained model path.")
    parser.add_argument("--text-column", default="text", help="Name of the text column in the input CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = load_classifier(args.model)
    output_path = predict_csv(args.input, args.output, classifier, text_column=args.text_column)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
