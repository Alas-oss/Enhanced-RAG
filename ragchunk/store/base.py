from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from ..types import Chunk

class VectorStore(ABC):
    @abstractmethod
    def add(Self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError