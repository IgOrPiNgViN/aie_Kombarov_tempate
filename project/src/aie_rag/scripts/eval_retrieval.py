from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from aie_rag.data.kb import chunk_text, load_kb
from aie_rag.retrieval.bm25 import bm25_search, build_bm25_index
from aie_rag.retrieval.faiss_index import build_faiss_index, build_index_in_memory, load_index, search


def hit_at_k(relevant: set[str], retrieved: list[str]) -> int:
    return int(any(x in relevant for x in retrieved))


def mrr_at_k(relevant: set[str], retrieved: list[str]) -> float:
    for i, x in enumerate(retrieved, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def run_evaluation(
    *,
    kb_path: Path,
    artifacts_dir: Path,
    benchmark_path: Path | None = None,
    chunk_size: int = 320,
    chunk_overlap: int = 60,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    retriever: str = "dense",
    top_k: int = 5,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Оценка retrieval на benchmark. Возвращает (таблица по запросам, сводка метрик)."""
    benchmark_path = benchmark_path or (artifacts_dir / "benchmark_queries.json")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    retriever = retriever.strip().lower()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    docs = load_kb(kb_path)
    chunks = chunk_text(docs=docs, chunk_size=chunk_size, overlap=chunk_overlap)

    if save_artifacts:
        build_faiss_index(chunks=chunks, embedding_model=embedding_model, artifacts_dir=artifacts_dir)
        index, chunk_meta = load_index(artifacts_dir=artifacts_dir)
    else:
        # Индекс в памяти: не пишем faiss на диск (обход Unicode-путей Windows в ноутбуке)
        index, chunk_meta = build_index_in_memory(chunks=chunks, embedding_model=embedding_model)

    bm25_idx = build_bm25_index(chunks=chunk_meta)

    rows: list[dict] = []
    for item in benchmark:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])
        if retriever == "bm25":
            res = bm25_search(idx=bm25_idx, query=query, top_k=top_k)
        elif retriever == "hybrid":
            dense = search(
                index=index,
                chunks=chunk_meta,
                query=query,
                embedding_model=embedding_model,
                top_k=top_k * 2,
            )
            lex = bm25_search(idx=bm25_idx, query=query, top_k=top_k * 2)
            score_map: dict[str, float] = {}
            for rank, r in enumerate(dense, start=1):
                score_map[r["chunk_id"]] = score_map.get(r["chunk_id"], 0.0) + (1.0 / rank)
            for rank, r in enumerate(lex, start=1):
                score_map[r["chunk_id"]] = score_map.get(r["chunk_id"], 0.0) + (1.0 / rank)
            merged = {r["chunk_id"]: r for r in (dense + lex)}
            res = sorted(
                merged.values(),
                key=lambda r: score_map.get(r["chunk_id"], 0.0),
                reverse=True,
            )[:top_k]
            for r in res:
                r["score"] = float(score_map.get(r["chunk_id"], 0.0))
        else:
            res = search(
                index=index,
                chunks=chunk_meta,
                query=query,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        retrieved_doc_ids = [r["doc_id"] for r in res]
        rows.append(
            {
                "query": query,
                "query_type": item.get("query_type", ""),
                "relevant_doc_ids": sorted(relevant),
                "retrieved_doc_ids": retrieved_doc_ids,
                "hit@5": hit_at_k(relevant, retrieved_doc_ids),
                "mrr@5": mrr_at_k(relevant, retrieved_doc_ids),
            }
        )

    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "hit@5": float(df["hit@5"].mean()),
        "mrr@5": float(df["mrr@5"].mean()),
        "retriever": retriever,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    if save_artifacts:
        df.to_csv(artifacts_dir / "retrieval_eval.csv", index=False, encoding="utf-8")
        (artifacts_dir / "retrieval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return df, summary


def main() -> None:
    from aie_rag.core.settings import get_settings

    s = get_settings()
    retriever = os.environ.get("AIE_RAG_RETRIEVER", "dense").strip().lower()
    _, summary = run_evaluation(
        kb_path=s.kb_path,
        artifacts_dir=s.artifacts_dir,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        embedding_model=s.embedding_model,
        retriever=retriever,
        save_artifacts=True,
    )
    print("Saved:")
    print(f"- {s.artifacts_dir / 'retrieval_eval.csv'}")
    print(f"- {s.artifacts_dir / 'retrieval_summary.json'}")
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
