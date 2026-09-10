from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_missing_question():
    response = client.post("/predict", json={"question1": "Only one question"})

    assert response.status_code == 422


def test_predict_rejects_blank_question():
    response = client.post(
        "/predict",
        json={"question1": "   ", "question2": "A valid question"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Both questions must be non-empty."}


def test_predict_returns_model_result(monkeypatch):
    def fake_predict(question1: str, question2: str) -> dict:
        assert question1 == "How do I learn Python?"
        assert question2 == "What is the best way to learn Python?"
        return {"is_duplicate": True, "probability": 0.91}

    monkeypatch.setattr(api, "predict", fake_predict)

    response = client.post(
        "/predict",
        json={
            "question1": "How do I learn Python?",
            "question2": "What is the best way to learn Python?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"is_duplicate": True, "probability": 0.91}