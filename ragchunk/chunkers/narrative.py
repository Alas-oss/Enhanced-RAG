from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
CHAPTER_PATTERN = re.compile(r"(?mi)^\s*chapter\s+\d+.*$")


class NarrativeProseChunker(BaseChunker):
    doc_type_label = "narrative_prose"

    def __init__(self, max_tokens: int = 600, overlap_tokens: int = 60):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def split(self, document: Document) -> List[dict]:
        text = document.text
        chapter_matches = list(CHAPTER_PATTERN.finditer(text))

        if chapter_matches:
            sections = []
            for i, m in enumerate(chapter_matches):
                start = m.start()
                end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
                sections.append((m.group(0).strip(), text[start:end]))
        else:
            sections = [(None, text)]

        pieces = []
        for chapter_label, section_text in sections:
            pieces.extend(self._pack_paragraphs(section_text, chapter_label))
        return pieces

    def _pack_paragraphs(self, text: str, section_path) -> List[dict]:
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
        pieces = []
        buffer_paragraphs: List[str] = []
        buffer_tokens = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            if buffer_tokens + para_tokens > self.max_tokens and buffer_paragraphs:
                pieces.append({"text": "\n\n".join(buffer_paragraphs), "section_path": section_path})
                buffer_paragraphs = [buffer_paragraphs[-1]] if self.overlap_tokens > 0 else []
                buffer_tokens = estimate_tokens(buffer_paragraphs[0]) if buffer_paragraphs else 0

            buffer_paragraphs.append(para)
            buffer_tokens += para_tokens

        if buffer_paragraphs:
            pieces.append({"text": "\n\n".join(buffer_paragraphs), "section_path": section_path})

        return pieces
