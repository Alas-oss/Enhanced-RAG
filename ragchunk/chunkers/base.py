from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import Chunk, Document
from ..utils import estimate_tokens

class BaseChunker(ABC):
    doc_type_label: str = "default"

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def split(self, document: Document) -> List[dict]:
        raise NotImplementedError

    def chunk(self, document: Document) -> List[Chunk]:
        raw_pieces = self.split(document)
        chunks: List[Chunk] = []
        for idx, piece in enumerate(raw_pieces):
            text = piece["text"].strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=f"{document.doc_id}::chunk_{idx}",
                    doc_id=document.doc_id,
                    doc_type=self.doc_type_label,
                    section_path=piece.get("section_path"),
                    chunk_index=idx,
                    token_estimate=estimate_tokens(text),
                    extra_metadata=piece.get("extra_metadata", {}),
                )
            )
        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        max_words = max(20, int(self.max_tokens / 1.3))
        overlap_words = max(0, int(self.overlap_tokens / 1.3))
        
        pieces = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            pieces.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap_words
        return pieces
        