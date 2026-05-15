# Docker

This project includes a simple Dockerfile for running the FastAPI service.

The image trains the baseline model during build using `data/sample_tickets.csv`. This keeps the demo self-contained and avoids committing generated model artifacts.

## Build

```bash
docker build -t ai-support-ticket-classifier .
```

## Run

```bash
docker run --rm -p 8000:8000 ai-support-ticket-classifier
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Test the API

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I was charged twice this month\"}"
```

## Limitations

- The image trains from the sample dataset during build.
- The model is still a demo baseline.
- For a real deployment, training and serving should be separated.
- Generated model artifacts are not committed to Git.
