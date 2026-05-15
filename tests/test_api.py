from fastapi.testclient import TestClient

from ticket_classifier.api.main import app


class FakeClassifier:
    def predict(self, text: str) -> dict[str, object]:
        return {
            "category": "billing",
            "confidence": 0.82,
            "probabilities": {
                "billing": 0.82,
                "sales": 0.08,
                "technical_support": 0.1,
            },
        }


def override_classifier() -> FakeClassifier:
    return FakeClassifier()


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_category() -> None:
    from ticket_classifier.model_service import get_classifier

    app.dependency_overrides[get_classifier] = override_classifier
    client = TestClient(app)

    response = client.post("/predict", json={"text": "I was charged twice"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["category"] == "billing"
    assert response.json()["confidence"] == 0.82


def test_predict_endpoint_validates_short_text() -> None:
    from ticket_classifier.model_service import get_classifier

    app.dependency_overrides[get_classifier] = override_classifier
    client = TestClient(app)

    response = client.post("/predict", json={"text": "hi"})

    app.dependency_overrides.clear()

    assert response.status_code == 422
