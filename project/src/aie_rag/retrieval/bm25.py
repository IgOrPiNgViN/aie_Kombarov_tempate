from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    # Учебная токенизация: достаточно для корпуса из коротких статей.
    return [t for t in text.lower().replace("\n", " ").split(" ") if t]


@dataclass(frozen=True)
class BM25Index:
    bm25: BM25Okapi
    chunks: list[dict]


def build_bm25_index(*, chunks: list[dict]) -> BM25Index:
    tokenized = [_tokenize(c.get("text", "")) for c in chunks]
    return BM25Index(bm25=BM25Okapi(tokenized), chunks=chunks)


def bm25_search(*, idx: BM25Index, query: str, top_k: int) -> list[dict]:
    tokens = _tokenize(query)
    scores = idx.bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]

    out: list[dict] = []
    for i in order:
        c = idx.chunks[i]
        out.append(
            {
                "doc_id": c["doc_id"],
                "title": c.get("title", ""),
                "score": float(scores[i]),
                "chunk_text": c.get("text", ""),
                "chunk_id": c.get("chunk_id", ""),
            }
        )
    return out

