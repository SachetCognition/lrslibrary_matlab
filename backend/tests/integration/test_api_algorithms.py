"""Integration tests for algorithm API endpoints."""


def test_list_all_algorithms(api_client):
    """GET /api/algorithms returns 200 with FPCP and IALM."""
    response = api_client.get("/api/algorithms")
    assert response.status_code == 200
    data = response.json()
    algo_ids = {a["algorithm_id"] for a in data}
    assert "FPCP" in algo_ids
    assert "IALM" in algo_ids


def test_list_by_method(api_client):
    """GET /api/algorithms/RPCA returns only RPCA algorithms."""
    response = api_client.get("/api/algorithms/RPCA")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    for algo in data:
        assert algo["method_id"] == "RPCA"


def test_invalid_method(api_client):
    """GET /api/algorithms/INVALID returns empty list."""
    response = api_client.get("/api/algorithms/INVALID")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_health_check(api_client):
    """GET /api/health returns ok."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_methods(api_client):
    """GET /api/methods returns list with RPCA."""
    response = api_client.get("/api/methods")
    assert response.status_code == 200
    data = response.json()
    assert "RPCA" in data
