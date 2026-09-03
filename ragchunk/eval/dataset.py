from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class LabeledExample:
    path: str
    expected_type: Optional[str] = None
    is_mixed: bool = False
    expected_section_types: Optional[List[str]] = None
    notes: Optional[str] = None

@dataclass
class QAExample:
    path: str
    question: str
    expected_section_contains: str
    doc_type: str
    top_k: int = 3

def load_labeled_dataset(path: str) -> List[LabeledExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LabeledExample(**item) for item in data]

def load_qa_dataset(path: str) -> List[QAExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAExample(**item) for item in data]
