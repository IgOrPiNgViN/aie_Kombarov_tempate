from __future__ import annotations

import json
from pathlib import Path

from aie_rag.core.logging import configure_logging
from aie_rag.core.settings import get_settings
from aie_rag.data.kb import chunk_text, load_kb
from aie_rag.retrieval.faiss_index import build_faiss_index


def main() -> None:
    s = get_settings()
    configure_logging(s.log_level)

    docs = load_kb(s.kb_path)
    chunks = chunk_text(docs=docs, chunk_size=s.chunk_size, overlap=s.chunk_overlap)
    artifacts = build_faiss_index(
        chunks=chunks,
        embedding_model=s.embedding_model,
        artifacts_dir=s.artifacts_dir,
    )

    # доп. метаданные, чтобы на защите было видно параметры индексации
    (s.artifacts_dir / "index_params.json").write_text(
        json.dumps(
            {
                "kb_path": str(s.kb_path),
                "embedding_model": s.embedding_model,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Built artifacts:")
    print(f"- {artifacts.index_path}")
    print(f"- {artifacts.chunks_path}")
    print(f"- {artifacts.meta_path}")


if __name__ == "__main__":
    main()

