from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    title: str
    text: str


def load_kb(path: Path) -> list[KBDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: list[KBDocument] = []
    for item in raw:
        docs.append(
            KBDocument(
                doc_id=str(item["doc_id"]),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
            )
        )
    return docs


@dataclass(frozen=True)
class KBChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int


def chunk_text(
    *,
    docs: list[KBDocument],
    chunk_size: int,
    overlap: int,
) -> list[KBChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: list[KBChunk] = []
    for d in docs:
        s = 0
        text = d.text or ""
        while s < len(text):
            e = min(len(text), s + chunk_size)
            ctext = text[s:e].strip()
            if ctext:
                chunk_id = f"{d.doc_id}::{s}-{e}"
                chunks.append(
                    KBChunk(
                        chunk_id=chunk_id,
                        doc_id=d.doc_id,
                        title=d.title,
                        text=ctext,
                        start=s,
                        end=e,
                    )
                )
            if e == len(text):
                break
            s = e - overlap
    return chunks


def chunks_to_jsonable(chunks: list[KBChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "text": c.text,
            "start": c.start,
            "end": c.end,
        }
        for c in chunks
    ]

