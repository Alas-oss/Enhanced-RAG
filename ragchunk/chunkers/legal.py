from __future__ import annotations

import re
from typing import List

from ..types import Document
from .base import BaseChunker

SECTION_HEADER_PATTERN = re.compile(
    r"""(?mx)
    ^\s*(
            ARTICLE\s+[IVXLC\d]+ .*|
            Section\s+\d+(\.\d+)* .*|
            \d+\.\d+(\.\d+)?\s+[A-Z][^\n]{0,80}|
            WHEREAS[,:]?
    )
    """
)

class LegalContractChunker(BaseChunker):
    doc_type_label = "legal_contract"

    def split(self, document: Document) -> List[dict]:
        text = document.text
        matches = list(SECTION_HEADER_PATTERN.finditer(text))

        if not matches:
            return [{"text": t, "section_path": None} for t in self._split_long_text(text)]

        pieces = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1< len(matches) else len(text)
            section_text = text[start:end].strip()
            section_label = match.group(1).strip()[:80]

            if not section_text:
                continue

            if self._token_len(section_text) <= self.max_tokens:
                pieces.append({"text": section_text, "section_path": section_label})
            else:
                for sub in self._split_long_text(section_text):
                    pieces.append({"text": sub, "section_path": section_label})

        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                pieces.insert(0, {"text": preamble, "section_path": "Preamble"})

        return pieces

    def _token_len(self, text: str) -> int:
        from ..utils import estimate_tokens

        return estimate_tokens(text)