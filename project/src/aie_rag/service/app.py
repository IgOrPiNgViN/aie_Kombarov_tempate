from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from aie_rag.core.logging import configure_logging
from aie_rag.core.settings import Settings, get_settings
from aie_rag.retrieval.faiss_index import load_index, search


REQUESTS_TOTAL = Counter("aie_rag_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("aie_rag_request_latency_seconds", "Request latency", ["endpoint"])

class Utf8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    doc_id: str
    title: str
    score: float
    chunk_id: str
    chunk_text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    meta: dict[str, Any]


@lru_cache(maxsize=1)
def _settings() -> Settings:
    s = get_settings()
    configure_logging(s.log_level)
    return s


@lru_cache(maxsize=1)
def _artifacts() -> tuple[Any, list[dict]]:
    s = _settings()
    return load_index(artifacts_dir=s.artifacts_dir)


app = FastAPI(
    title="Campus Search API",
    version="0.3.0",
    description="Семантический поиск по корпусу (dense retrieval + FAISS). Документация: GET /docs",
    default_response_class=Utf8JSONResponse,
)


@app.get("/")
def root() -> dict[str, str]:
    """Краткая справка по API (без веб-UI)."""
    return {
        "service": "Campus Search",
        "docs": "/docs",
        "health": "/health",
        "search": "POST /search",
        "predict": "POST /predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    s = _settings()
    ok = True
    missing: list[str] = []
    for name in ["faiss.index", "chunks.json", "meta.json"]:
        if not (s.artifacts_dir / name).exists():
            ok = False
            missing.append(name)
    return {
        "status": "ok" if ok else "degraded",
        "artifacts_dir": str(s.artifacts_dir),
        "missing_artifacts": missing,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _run_search(req: SearchRequest, *, endpoint: str) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        s = _settings()
        index, chunks = _artifacts()
        hits = search(
            index=index,
            chunks=chunks,
            query=req.query,
            embedding_model=s.embedding_model,
            top_k=req.top_k,
        )
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="200").inc()
        return SearchResponse(
            query=req.query,
            results=[SearchHit(**x) for x in hits],
            meta={
                "top_k": req.top_k,
                "embedding_model": s.embedding_model,
                "retriever": "dense",
            },
        )
    except FileNotFoundError as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="503").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                "Index artifacts are missing. Build them with "
                "`python -m aie_rag.scripts.build_index` and retry."
            ),
        ) from e
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)


@app.post("/search", response_model=SearchResponse)
def search_documents(req: SearchRequest) -> SearchResponse:
    """Семантический поиск top-k фрагментов."""
    return _run_search(req, endpoint="/search")


@app.post("/predict", response_model=SearchResponse)
def predict(req: SearchRequest) -> SearchResponse:
    """Тот же поиск (контракт сквозного проекта)."""
    return _run_search(req, endpoint="/predict")
