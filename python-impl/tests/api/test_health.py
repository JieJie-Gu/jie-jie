from fastapi.testclient import TestClient
from smart_cs.main import app


def test_health_reports_foundation_phase() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "smart-cs-agent",
        "phase": "foundation",
    }
