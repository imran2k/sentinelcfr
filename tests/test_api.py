import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ask_endpoint():
    response = client.get("/ask", params={"q": "head protection requirements"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)

def test_report_endpoint():
    violations = [{"person": 0, "missing": "helmet", "box": [10, 10, 50, 100]}]
    response = client.post("/report", json=violations)
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert isinstance(data["report"], str)
    assert len(data["report"].strip()) > 0
