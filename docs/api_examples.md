# API Examples

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

## Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Can you send me the invoice for my last payment?\"}"
```

Response:

```json
{
  "category": "billing",
  "confidence": 0.34,
  "probabilities": {
    "appointment": 0.12,
    "billing": 0.34,
    "complaint": 0.11,
    "other": 0.13,
    "sales": 0.14,
    "technical_support": 0.16
  }
}
```

## Validation Error

Request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"hi\"}"
```

The API returns `422` because the message is shorter than the minimum accepted length.

## Batch Prediction CLI

```bash
predict-ticket-batch --input tickets.csv --output predictions.csv
```

The input CSV must contain a `text` column.
