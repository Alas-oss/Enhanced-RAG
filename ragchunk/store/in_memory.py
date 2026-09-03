from __future__ import annotations

import math
from typing import List, Tuple

from ..types import Chunk
from .base import VectorStore

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings must match one-to-one.")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        scored = [
            (chunk, _cosine(query_embedding, emb))
            for chunk, emb in zip(self._chunks, self._embeddings)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)