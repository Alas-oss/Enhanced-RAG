from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

HEADING_PATTERN = re.compile(r"(?m)^(#{1,4})\s+(.+)$")

TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$")

class FinancialReportChunker(BaseChunker):
    doc_type_label = "financial_report"

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def split(self, document: Document) -> List[dict]:
        text = document.text
        headings = list(HEADING_PATTERN.finditer(text))

        if not headings:
            return self._split_section(text, section_path=None)

        pieces = []
        path_stack: List[str] = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end]

            level = len(match.group(1))
            title = match.group(2).strip()
            path_stack = path_stack[: level - 1] + [title]
            section_path = " > ".join(path_stack)

            pieces.extend(self._split_section(section_text, section_path))

        return pieces

    def _split_section(self, text: str, section_path) -> List[dict]:
        lines = text.split("\n")
        pieces = []
        buffer: List[str] = []
        i = 0

        while i < len(lines):
            if TABLE_ROW_PATTERN.match(lines[i]):
                if buffer:
                    pieces.extend(self._pack_narrative("\n".join(buffer), section_path))
                    buffer = []

                table_lines = []
                while i < len(lines) and (TABLE_ROW_PATTERN.match(lines[i]) or lines[i].strip() == ""):
                    if lines[i].strip():
                        table_lines.append(lines[i])
                    i += 1

                table_text = "\n".join(table_lines).strip()
                if table_text:
                    pieces.append({
                        "text": table_text,
                        "section_path": section_path,
                        "extra_metadata": {"content_type": "table"},
                    })
                continue

            buffer.append(lines[i])
            i += 1

        if buffer:
            pieces.extend(self._pack_narrative("\n".join(buffer), section_path))

        return pieces

    def _pack_narrative(self, text: str, section_path) -> List[dict]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        pieces = []
        buffer: List[str] = []
        buffer_tokens = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            if buffer_tokens + para_tokens > self.max_tokens and buffer:
                pieces.append({"text": "\n\n".join(buffer), "section_path": section_path})
                buffer, buffer_tokens = [], 0
            buffer.append(para)
            buffer_tokens += para_tokens

        if buffer:
            pieces.append({"text": "\n\n".join(buffer), "section_path": section_path})

        return pieces