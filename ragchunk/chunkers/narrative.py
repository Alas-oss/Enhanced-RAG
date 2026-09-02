from __future__ import annotations

import math
import re
from typing import Callable, List, Optional

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
CHAPTER_PATTERN = re.compile(r"(?mi)^\s*chapter\s+\d+.*$")

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class NarrativeProseChunker(BaseChunker):
    doc_type_label = "narrative_prose"

    def __init__(self, 
            max_tokens: int = 600, 
            overlap_tokens: int = 60,
            embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
            breakpoint_percentile: float = 0.25,
            ):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile

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
            if self.embed_fn is not None:
                pieces.extend(self._semantic_pack(section_text, chapter_label))
            else:
                pieces.extend(self._pack_paragraphs(section_text, chapter_label))
        return pieces

    def _semantic_pack(self, text: str, section_path) -> List[dict]:
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
        if len(paragraphs) <= 1:
            return [{"text": text.strip(), "section_path": section_path}] if text.strip() else []

        try:
            embeddings = self.embed_fn(paragraphs)
        except Exception:
            return self._pack_paragraphs(text, section_path)

        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        if not similarities:
            return self._pack_paragraphs(text, section_path)

        sorted_sims = sorted(similarities)
        idx = min(int(len(sorted_sims) * self.breakpoint_percentile), len(sorted_sims) - 1)
        threshold = sorted_sims[idx]

        groups: List[List[str]] = [{paragraphs[0]}]
        for i, sim in enumerate(similarities):
            if sim <= threshold:
                groups.append({paragraphs[i + 1]})
            else:
                groups[-1].append(paragraphs[i + 1])

        pieces = []
        for group in groups:
            group_text = "\n\n".join(group)
            if estimate_tokens(group_text) <= self.max_tokens:
                pieces.append({"text": group_text, "section_path": section_path})
            else:
                pieces.extend(self._pack_paragraphs(group_text, section_path))
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
