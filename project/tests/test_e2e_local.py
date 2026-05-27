from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _clear_app_caches() -> None:
    # the app module caches settings and artifacts in lru_cache
    from aie_rag.service import app as app_module

    app_module._settings.cache_clear()
    app_module._artifacts.cache_clear()


def test_api_root_has_endpoints() -> None:
    from aie_rag.service.app import app

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


def test_metrics_endpoint_works() -> None:
    from aie_rag.service.app import app

    c = TestClient(app)
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "aie_rag_requests_total" in r.text


def test_search_returns_503_if_no_artifacts(tmp_artifacts: Path) -> None:
    from aie_rag.service.app import app

    _clear_app_caches()
    c = TestClient(app)
    r = c.post("/search", json={"query": "справка", "top_k": 3})
    assert r.status_code == 503
    assert "Index artifacts are missing" in r.text


def test_build_index_creates_artifacts_and_search_works(
    built_artifacts: Path,
) -> None:
    from aie_rag.service.app import app

    _clear_app_caches()
    c = TestClient(app)
    r = c.post("/search", json={"query": "как получить справку об обучении?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["results"], "Expected non-empty results"
    assert any(x["doc_id"] == "campus_01" for x in body["results"])


def test_predict_is_alias_of_search(built_artifacts: Path) -> None:
    from aie_rag.service.app import app

    _clear_app_caches()

    c = TestClient(app)
    a = c.post("/search", json={"query": "справка", "top_k": 3}).json()
    b = c.post("/predict", json={"query": "справка", "top_k": 3}).json()
    assert a["results"][0]["doc_id"] == b["results"][0]["doc_id"]

