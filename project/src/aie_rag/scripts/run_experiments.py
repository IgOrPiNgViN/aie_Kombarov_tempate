"""Полный цикл offline-экспериментов: матрица конфигураций + разбивка по запросам + графики."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aie_rag.core.logging import configure_logging
from aie_rag.core.settings import get_settings
from aie_rag.scripts.eval_retrieval import run_evaluation

CHUNK_SIZES = (200, 320)
CHUNK_OVERLAPS = (40, 60)
RETRIEVERS = ("dense", "bm25", "hybrid")


def run_experiment_matrix(
    *,
    kb_path: Path,
    artifacts_dir: Path,
    benchmark_path: Path,
    embedding_model: str,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk_size in CHUNK_SIZES:
        for overlap in CHUNK_OVERLAPS:
            for retriever in RETRIEVERS:
                _, summary = run_evaluation(
                    kb_path=kb_path,
                    artifacts_dir=artifacts_dir,
                    benchmark_path=benchmark_path,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    embedding_model=embedding_model,
                    retriever=retriever,
                    top_k=top_k,
                    save_artifacts=False,
                )
                rows.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "retriever": retriever,
                        "hit@5": summary["hit@5"],
                        "mrr@5": summary["mrr@5"],
                        "num_queries": None,
                    }
                )
    df = pd.DataFrame(rows)
    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    df["num_queries"] = len(bench)
    return df


def run_per_query_breakdown(
    *,
    kb_path: Path,
    artifacts_dir: Path,
    benchmark_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    retriever: str,
    top_k: int = 5,
) -> pd.DataFrame:
    df, _ = run_evaluation(
        kb_path=kb_path,
        artifacts_dir=artifacts_dir,
        benchmark_path=benchmark_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        retriever=retriever,
        top_k=top_k,
        save_artifacts=False,
    )
    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    qtype = {item["query"]: item.get("query_type", "unknown") for item in bench}
    df["query_type"] = df["query"].map(qtype)
    df["retriever"] = retriever
    df["chunk_size"] = chunk_size
    df["chunk_overlap"] = chunk_overlap
    return df


def _save_figures(matrix: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    dense = matrix[matrix["retriever"] == "dense"].copy()
    if not dense.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{int(r.chunk_size)}/{int(r.chunk_overlap)}" for r in dense.itertuples()]
        ax.bar(labels, dense["mrr@5"].tolist(), color="steelblue")
        ax.set_title("Dense: MRR@5 по chunk_size / overlap")
        ax.set_ylabel("mrr@5")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "mrr_dense_chunk_overlap.png", dpi=120)
        plt.close(fig)

    best_overlap = 60
    sub = matrix[matrix["chunk_overlap"] == best_overlap].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for retriever in RETRIEVERS:
            part = sub[sub["retriever"] == retriever]
            if part.empty:
                continue
            ax.plot(
                part["chunk_size"].astype(str),
                part["mrr@5"],
                marker="o",
                label=retriever,
            )
        ax.set_title(f"MRR@5: retriever (overlap={best_overlap})")
        ax.set_xlabel("chunk_size")
        ax.set_ylabel("mrr@5")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "mrr_by_retriever.png", dpi=120)
        plt.close(fig)

    sub320 = matrix[(matrix["chunk_size"] == 320) & (matrix["chunk_overlap"] == 60)]
    if not sub320.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sub320["retriever"], sub320["mrr@5"], color=["#2a6f97", "#e9c46a", "#e76f51"])
        ax.set_title("MRR@5 по retriever (320/60)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "mrr_retriever_320_60.png", dpi=120)
        plt.close(fig)


def main() -> None:
    s = get_settings()
    configure_logging(s.log_level)
    artifacts = s.artifacts_dir
    benchmark = artifacts / "benchmark_queries.json"
    figures = artifacts / "figures"

    print("Running experiment matrix (chunk x overlap x retriever)...")
    matrix = run_experiment_matrix(
        kb_path=s.kb_path,
        artifacts_dir=artifacts,
        benchmark_path=benchmark,
        embedding_model=s.embedding_model,
    )
    matrix_path = artifacts / "retrieval_experiment_matrix.csv"
    matrix.to_csv(matrix_path, index=False, encoding="utf-8")
    print(f"Saved: {matrix_path}")

    print("Per-query breakdown (production config: dense, 320/60)...")
    per_query = run_per_query_breakdown(
        kb_path=s.kb_path,
        artifacts_dir=artifacts,
        benchmark_path=benchmark,
        chunk_size=320,
        chunk_overlap=60,
        embedding_model=s.embedding_model,
        retriever="dense",
    )
    per_query_path = artifacts / "retrieval_per_query_dense_320_60.csv"
    per_query.to_csv(per_query_path, index=False, encoding="utf-8")
    print(f"Saved: {per_query_path}")

    for retriever in RETRIEVERS:
        pq = run_per_query_breakdown(
            kb_path=s.kb_path,
            artifacts_dir=artifacts,
            benchmark_path=benchmark,
            chunk_size=320,
            chunk_overlap=60,
            embedding_model=s.embedding_model,
            retriever=retriever,
        )
        path = artifacts / f"retrieval_per_query_{retriever}_320_60.csv"
        pq.to_csv(path, index=False, encoding="utf-8")

    summary_rows = []
    for retriever in RETRIEVERS:
        part = matrix[(matrix["chunk_size"] == 320) & (matrix["chunk_overlap"] == 60)]
        row = part[part["retriever"] == retriever]
        if not row.empty:
            summary_rows.append(row.iloc[0].to_dict())
    summary_df = pd.DataFrame(summary_rows)
    summary_path = artifacts / "retrieval_experiments_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved: {summary_path}")

    _save_figures(matrix, figures)
    if (figures / "mrr_dense_chunk_overlap.png").exists():
        print(f"Saved figures in: {figures}")

    print("\nBest dense configs by mrr@5:")
    dense = matrix[matrix["retriever"] == "dense"].sort_values("mrr@5", ascending=False)
    print(dense.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
