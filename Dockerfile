FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models/.gitkeep ./models/.gitkeep

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e "."

RUN train-ticket-classifier

EXPOSE 8000

CMD ["uvicorn", "ticket_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
