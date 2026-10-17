from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


def test_liveness_returns_200():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert "status" in response.json()


def test_readiness_returns_200():
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert "status" in response.json()