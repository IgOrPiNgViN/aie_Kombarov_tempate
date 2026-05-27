from __future__ import annotations

import sys
from pathlib import Path

import pytest


def pytest_configure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class _FakeEmbedder:
    """Tiny deterministic embedder for tests (no HF/torch download)."""

    def encode(self, texts: list[str], **_):  # sentence-transformers compatible
        import numpy as np

        out = []
        for t in texts:
            s = (t or "").lower()
            # very simple "semantic": справка-related texts map to axis 0
            if "справк" in s:
                v = np.array([1.0, 0.0], dtype="float32")
            else:
                v = np.array([0.0, 1.0], dtype="float32")
            out.append(v)
        return np.stack(out, axis=0)


@pytest.fixture()
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> _FakeEmbedder:
    from aie_rag.retrieval import faiss_index

    # clear lru_cache on the real function before replacing it
    faiss_index._get_embedder.cache_clear()

    emb = _FakeEmbedder()
    monkeypatch.setattr(faiss_index, "_get_embedder", lambda _: emb)
    return emb


@pytest.fixture()
def tmp_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    # configure settings via env
    monkeypatch.setenv("AIE_RAG_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("AIE_RAG_KB_PATH", str(Path("data/knowledge_base.json").resolve()))
    monkeypatch.setenv("AIE_RAG_EMBEDDING_MODEL", "fake-embedder")
    monkeypatch.setenv("AIE_RAG_CHUNK_SIZE", "320")
    monkeypatch.setenv("AIE_RAG_CHUNK_OVERLAP", "60")
    return tmp_path / "artifacts"


@pytest.fixture()
def built_artifacts(fake_embedder: _FakeEmbedder, tmp_artifacts: Path) -> Path:
    from aie_rag.scripts import build_index

    build_index.main()
    assert (tmp_artifacts / "faiss.index").exists()
    assert (tmp_artifacts / "chunks.json").exists()
    assert (tmp_artifacts / "meta.json").exists()
    return tmp_artifacts

