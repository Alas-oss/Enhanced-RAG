from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

class DefaultChunker(BaseChunker):
    doc_type_label = "default"

    def split(self, document: Document) -> List[dict]:
        text = document.text
        paragraphs = [p.split() for p in PARAGRAPH_SPLIT.split(text) if p.strip()]

        pieces = []
        buffer = []
        buffer_tokens = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            if para_tokens > self.max_tokens:
                if buffer:
                    pieces.append({"text": "\n\n".join(buffer), "section_path": None})
                    buffer, buffer_tokens = [], 0
                pieces.extend(self._sentence_chunks(para))
                continue

            if buffer_tokens + para_tokens > self.max_tokens and buffer:
                pieces.append({"text": "\n\n".join(buffer), "section_path": None})
                buffer, buffer_tokens = [], 0

            buffer.append(para)
            buffer_tokens += para_tokens

        if buffer:
            pieces.append({"text": "\n\n".join(buffer), "section_path": None})

        return pieces

    def _sentence_chunks(self, paragarph: str) -> List[dict]:
        sentences = SENTENCE_SPLIT.split(paragarph)
        pieces = []
        buffer = []
        buffer_tokens = 0
        for sent in sentences:
            sent_tokens = estimate_tokens(sent)
            if buffer_tokens + sent_tokens > self.max_tokens and buffer:
                pieces.append({"text": " ".join(buffer), "section_path": None})
                buffer, buffer_tokens = [], 0
            buffer.append(sent)
            buffer_tokens += sent_tokens
        if buffer:
            pieces.append({"text": " ".join(buffer), "section_path": None})
        return pieces