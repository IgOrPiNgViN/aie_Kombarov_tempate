from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from aie_rag.data.kb import KBChunk, chunks_to_jsonable


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@lru_cache(maxsize=4)
def _get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@dataclass(frozen=True)
class IndexArtifacts:
    index_path: Path
    chunks_path: Path
    meta_path: Path


def _encode_chunks(chunks: list[KBChunk], embedding_model: str) -> tuple[faiss.Index, np.ndarray]:
    model = _get_embedder(embedding_model)
    texts = [c.text for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    emb = emb.astype("float32")
    emb = _l2_normalize(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def _write_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """Запись индекса; обход ошибки FAISS на Windows с путями в Unicode."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(index_path))
        return
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="faiss_") as tmp:
            tmp_file = Path(tmp) / "faiss.index"
            faiss.write_index(index, str(tmp_file))
            shutil.copy2(tmp_file, index_path)


def build_index_in_memory(
    *,
    chunks: list[KBChunk],
    embedding_model: str,
) -> tuple[faiss.Index, list[dict]]:
    """Индекс только в RAM — для экспериментов в ноутбуке без записи на диск."""
    index, _ = _encode_chunks(chunks, embedding_model)
    return index, chunks_to_jsonable(chunks)


def build_faiss_index(
    *,
    chunks: list[KBChunk],
    embedding_model: str,
    artifacts_dir: Path,
) -> IndexArtifacts:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index, _ = _encode_chunks(chunks, embedding_model)

    index_path = artifacts_dir / "faiss.index"
    chunks_path = artifacts_dir / "chunks.json"
    meta_path = artifacts_dir / "meta.json"

    _write_faiss_index(index, index_path)
    chunks_path.write_text(json.dumps(chunks_to_jsonable(chunks), ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "embedding_model": embedding_model,
        "num_chunks": len(chunks),
        "faiss": {"type": "IndexFlatIP", "metric": "cosine_via_l2_norm"},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return IndexArtifacts(index_path=index_path, chunks_path=chunks_path, meta_path=meta_path)


def load_index(*, artifacts_dir: Path) -> tuple[faiss.Index, list[dict]]:
    index_path = artifacts_dir / "faiss.index"
    chunks_path = artifacts_dir / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        missing = [p.name for p in [index_path, chunks_path] if not p.exists()]
        raise FileNotFoundError(f"Missing artifacts in {artifacts_dir}: {', '.join(missing)}")

    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="faiss_") as tmp:
            tmp_file = Path(tmp) / "faiss.index"
            shutil.copy2(index_path, tmp_file)
            index = faiss.read_index(str(tmp_file))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, chunks


def search(
    *,
    index: faiss.Index,
    chunks: list[dict],
    query: str,
    embedding_model: str,
    top_k: int,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    model = _get_embedder(embedding_model)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = _l2_normalize(q)

    scores, ids = index.search(q, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: list[dict] = []
    for score, idx in zip(scores, ids, strict=False):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append(
            {
                "doc_id": c["doc_id"],
                "title": c.get("title", ""),
                "score": float(score),
                "chunk_text": c.get("text", ""),
                "chunk_id": c.get("chunk_id", ""),
            }
        )
    return results

