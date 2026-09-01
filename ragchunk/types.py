from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class DocType(str, Enum):
    LEGAL_CONTRACT = "legal_contract"
    GOVERNMENT_REGULATION = "government_regulation"
    TECHNICAL_MANUAL = "technical_manual"
    FINANCIAL_REPORT = "financial_report"
    NARRATIVE_PROSE = "narrative_prose"
    DEFAULT = "default"

@dataclass
class ClassificationResult:
    doc_type: DocType
    confidence: float
    method: str
    signals: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    doc_type: str
    section_path: Optional[str] = None
    chunk_index: int = 0
    token_estimate: int = 0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "section_path": self.section_path,
            "chunk_index": self.chunk_index,
            "token_estimate": self.token_estimate,
            "text": self.text,
            "metadata": self.extra_metadata,
        }

@dataclass
class Document:
    doc_id: str
    text: str
    source_path: Optional[str] = None
    file_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult: 
    doc_id: str
    doc_type: DocType
    classification: ClassificationResult
    chunks: List[Chunk]
