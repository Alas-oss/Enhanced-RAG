from __future__ import annotations

from typing import Dict, Optional

from .chunkers import (
    BaseChunker,
    DefaultChunker,
    GovernmentRegulationChunker,
    LegalContractChunker,
    NarrativeProseChunker,
    TechnicalManualChunker,
)
from .types import DocType


class ChunkerRouter:
    def __init__(self, overrides: Optional[Dict[DocType, BaseChunker]] = None):
        self._registry: Dict[DocType, BaseChunker] = {
            DocType.LEGAL_CONTRACT: LegalContractChunker(max_tokens=300, overlap_tokens=50),
            DocType.GOVERNMENT_REGULATION: GovernmentRegulationChunker(max_tokens=250, overlap_tokens=30),
            DocType.TECHNICAL_MANUAL: TechnicalManualChunker(max_tokens=500, overlap_tokens=50),
            DocType.NARRATIVE_PROSE: NarrativeProseChunker(max_tokens=600, overlap_tokens=60),
            DocType.FINANCIAL_REPORT: DefaultChunker(max_tokens=400, overlap_tokens=50),
            DocType.DEFAULT: DefaultChunker(max_tokens=500, overlap_tokens=50),
        }
        if overrides:
            self._registry.update(overrides)

    def get_chunker(self, doc_type: DocType) -> BaseChunker:
        return self._registry.get(doc_type, self._registry[DocType.DEFAULT])

    def register(self, doc_type: DocType, chunker: BaseChunker) -> None:
        self._registry[doc_type] = chunker
