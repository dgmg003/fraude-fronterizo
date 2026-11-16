from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from frontera_fraud.api.main import app
from frontera_fraud.config import Settings
from frontera_fraud.ingestion.synthetic import generate_synthetic_transactions
from frontera_fraud.services.detection import FraudDetectionService


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        original_service = app.state.service
        try:
            yield test_client
        finally:
            app.state.service = original_service


@pytest.fixture(scope="module")
def trained_service() -> FraudDetectionService:
    cfg = Settings(anonymization_salt="api-test-salt", n_estimators=50)
    data = generate_synthetic_transactions(n_rows=1000, seed=123)
    return FraudDetectionService(cfg).train(data)


def sample_transaction() -> dict:
    return {
        "transaction_id": "T-PRUEBA-001",
        "merchant_id": "M-001",
        "counterparty_id": "C-001",
        "timestamp": "2025-01-10T10:30:00",
        "invoice_number": "F-001",
        "product_code": "P-001",
        "quantity": 2,
        "unit_price": 125000,
        "reference_unit_price": 130000,
        "amount_cop": 250000,
        "exchange_rate_applied": 100.2,
        "exchange_rate_official": 100.0,
        "payer_is_third_party": False,
    }


def test_health_endpoint_responds(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_loaded"], bool)


def test_score_rejects_empty_payload(client: TestClient):
    response = client.post("/v1/transactions/score", json={"transactions": []})

    assert response.status_code == 422


def test_score_returns_risk_result(
    client: TestClient,
    trained_service: FraudDetectionService,
):
    app.state.service = trained_service

    response = client.post(
        "/v1/transactions/score",
        json={"transactions": [sample_transaction()]},
    )

    assert response.status_code == 200
    body = response.json()
    assert "revisión humana" in body["disclaimer"]

    result = body["results"][0]
    assert result["transaction_id"] == "T-PRUEBA-001"
    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] in {"bajo", "medio", "alto"}
    assert result["requires_review"] == (result["risk_level"] != "bajo")


def test_score_returns_503_when_model_is_unavailable(client: TestClient):
    app.state.service = FraudDetectionService(
        Settings(anonymization_salt="api-test-salt")
    )

    response = client.post(
        "/v1/transactions/score",
        json={"transactions": [sample_transaction()]},
    )

    assert response.status_code == 503
