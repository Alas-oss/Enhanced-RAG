from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker
from ..utils import estimate_tokens

HEADER_PATTERN = re.compile(
    r"""(?mx)
    ^\s*(
        TITLE\s+[\dA-Z]+ .*|
        PART\s+\d+ .*|
        Subpart\s+[A-Z] .*|
        §\s?\d+(\.\d+)* .*|
        Sec(tion)?\.?\s+\d+(\.\d+)* .*
    )
    """
)

CITATIOM_PATTERN = re.compile(r"\b\d{1,3}\s*U\.?S\.?C\.?\s*§?\s*\d+|\bC\.?F\.?R\.?\s*§?\s*[\d.]+")

class GovernmentRegulationChunker(BaseChunker):
    doc_type_label = "government_regulation"

    def __init__(self, max_tokens = 250, overlap_tokens = 30):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def split(self, document: Document) -> List[dict]:
        text = document.text
        matches = list(HEADER_PATTERN.finditer(text))

        if not matches:
            return [{"text": t, "section_path": None} for t in self._split_long_text(text)]

        pieces = []
        path_stack: List[str] = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            header = match.group(1).strip()[:80]

            path_stack = self._update_path_stack(path_stack, header)
            section_path = " > ".join(path_stack)

            if not section_text:
                continue

            if estimate_tokens(section_text) < self.max_tokens:
                pieces.append({
                    "text": section_text,
                    "section_path": section_path,
                    "extra_metadata": {"contains_citation": bool(CITATIOM_PATTERN.search(section_text))}
                })
            else: 
                for sub in self._split_long_text(section_text):
                    pieces.append({
                        "text": sub,
                        "section_path": section_path,
                        "extra_metadata": {"contains_citation": bool(CITATIOM_PATTERN.search(sub))},
                    })
        return pieces

    @staticmethod
    def _update_path_stack(stack: List[str], header: str) -> List[str]:
        level = 0
        if header.upper().startswith("TITLE"):
            level = 0
        elif header.upper().startswith("PART"):
            start = 1
        elif header.upper().startswith("SUBPART"):
            level = 2
        else:
            level = 3

        new_stack = stack[:level]
        new_stack.append(header)
        return new_stack