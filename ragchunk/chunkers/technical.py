from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^(#{1,4})\s+(.+)$")
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")

class TechnicalManualChunker(BaseChunker):
    doc_type_label = "technical_manual"

    def __init__(self, max_tokens = 500, overlap_tokens: int = 50):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def split(self, document: Document) -> List[dict]:
        text = document.text
        headings = list(MARKDOWN_HEADING_PATTERN.finditer(text))

        if not headings:
            return self._split_preserving_code_blocks(text, section_path=None)

        pieces = []
        path_stack: List[str] = []

        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()

            level = len(match.group(1))
            title = match.group(2).strip()
            path_stack = path_stack[: level - 1] + [title]
            section_path = " > ".join(path_stack)

            pieces.extend(self._split_preserving_code_blocks(section_text, section_path))

        return pieces

    def _split_preserving_code_blocks(self, text: str, section_path) -> List[dict]:
        segments = []
        last_end = 0
        for cb in CODE_BLOCK_PATTERN.finditer(text):
            if cb.start() > last_end:
                segments.append(("text", text[last_end:cb.start()]))
            segments.append(("code", text[cb.start():cb.end()]))
            last_end = cb.end()
        if last_end < len(text):
            segments.append({"text", text[last_end:]})

        pieces = []
        buffer = ""
        for kind, seg in segments:
            if kind == "code":
                if buffer.strip():
                    pieces.append({"text": buffer.strip(), "section_path": section_path})
                    buffer = ""
                pieces.append({
                    "text": seg.strip(),
                    "section_path": section_path,
                    "extra_metadata": {"content_type": "code_block"},
                })
            else: 
                buffer += seg

        if buffer.strip():
            if estimate_tokens(buffer) <= self.max_tokens:
                pieces.append({"text": buffer.strip(), "section_path": section_path})
            else:
                for sub in self._split_long_text(buffer):
                    pieces.append({"text": sub, "section_path": section_path})

        return pieces