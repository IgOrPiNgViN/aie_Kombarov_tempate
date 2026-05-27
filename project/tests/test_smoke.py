from __future__ import annotations

from fastapi.testclient import TestClient

from aie_rag.service.app import app


def test_health_endpoint() -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_search_returns_results_when_artifacts_built(built_artifacts) -> None:
    c = TestClient(app)
    r = c.post("/search", json={"query": "что такое FAISS?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert body["query"]
    assert len(body["results"]) == 3


def test_predict_alias(built_artifacts) -> None:
    c = TestClient(app)
    r = c.post("/predict", json={"query": "chunking overlap", "top_k": 3})
    assert r.status_code == 200
