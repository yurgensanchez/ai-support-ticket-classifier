# AI Support Ticket Classifier

[![Tests](https://github.com/yurgensanchez/ai-support-ticket-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/yurgensanchez/ai-support-ticket-classifier/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![License](https://img.shields.io/badge/license-MIT-green)

FastAPI service that classifies short support messages into business-oriented categories.

This project is part of a professional portfolio focused on Python, applied machine learning, APIs, testing, and clear documentation. It is intentionally small: the goal is to show a complete baseline ML workflow, not to claim production-level model quality.

## Problem

Small businesses often receive customer messages that need to be routed to the right team. A simple classifier can help separate messages into categories such as sales, technical support, billing, complaints, appointments, or general questions.

## Current Categories

- `sales`
- `technical_support`
- `billing`
- `complaint`
- `appointment`
- `other`

## Approach

The first version uses a classic text classification baseline:

- text normalization
- TF-IDF features
- Logistic Regression classifier
- FastAPI endpoint for predictions
- metrics saved after training

This approach is easy to understand, fast to run locally, and appropriate for a small portfolio project.

## Project Structure

```text
ai-support-ticket-classifier/
├── data/
│   └── sample_tickets.csv
├── docs/
├── models/
├── src/
│   └── ticket_classifier/
│       ├── api/
│       ├── ml/
│       └── model_service.py
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Train the Model

```bash
train-ticket-classifier
```

This creates:

```text
models/ticket_classifier.joblib
models/metrics.json
models/metrics_summary.md
```

## Run the API

```bash
uvicorn ticket_classifier.api.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Predict Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I was charged twice this month\"}"
```

Example response:

```json
{
  "category": "billing",
  "confidence": 0.32,
  "probabilities": {
    "appointment": 0.12,
    "billing": 0.32,
    "complaint": 0.14,
    "other": 0.11,
    "sales": 0.16,
    "technical_support": 0.15
  }
}
```

The exact probabilities may change when the dataset or model changes.

## Batch Predictions

Create a CSV with a `text` column:

```csv
text
I was charged twice this month
Can I book a meeting for Friday?
```

Run batch prediction:

```bash
predict-ticket-batch --input tickets.csv --output predictions.csv
```

The output CSV includes the original text, predicted category, confidence, and per-class probabilities.

## Tests

```bash
pytest
```

## Documentation

- [Model Card](docs/model_card.md)
- [Dataset Notes](docs/dataset.md)
- [API Examples](docs/api_examples.md)

## Limitations

- The dataset is small and created for demonstration.
- The model should not be used for real customer routing without more data and evaluation.
- The classifier may confuse similar categories such as billing complaints and general complaints.
- Confidence scores are model probabilities, not a guarantee that the prediction is correct.

## Next Steps

- Expand the dataset with more realistic examples.
- Add confusion matrix reporting.
- Improve model evaluation with a larger dataset.
- Compare the baseline against embeddings or an LLM-based classifier.
- Add Docker support.
- Add GitHub Actions for automated tests.
