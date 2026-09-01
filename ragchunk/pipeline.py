from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, List, Optional

from .classifier import HybridClassifier
from .router import ChunkerRouter
from .types import Document, PipelineResult
from .utils import load_text_file, normalize_whitespace


class AdaptiveChunkingPipeline:
    def __init__(
        self,
        llm_classify_fn: Optional[Callable[[str], str]] = None,
        router: Optional[ChunkerRouter] = None,
    ):
        self.classifier = HybridClassifier(llm_classify_fn=llm_classify_fn)
        self.router = router or ChunkerRouter()

    def process_text(self, text: str, doc_id: Optional[str] = None, source_path: Optional[str] = None) -> PipelineResult:
        doc_id = doc_id or str(uuid.uuid4())
        clean_text = normalize_whitespace(text)
        document = Document(doc_id=doc_id, text=clean_text, source_path=source_path)

        classification = self.classifier.classify(clean_text)
        chunker = self.router.get_chunker(classification.doc_type)
        chunks = chunker.chunk(document)

        return PipelineResult(
            doc_id=doc_id,
            doc_type=classification.doc_type,
            classification=classification,
            chunks=chunks,
        )

    def process_file(self, path: str) -> PipelineResult:
        text = load_text_file(path)
        doc_id = Path(path).stem
        return self.process_text(text, doc_id=doc_id, source_path=path)

    def process_directory(self, dir_path: str, pattern: str = "*.txt") -> List[PipelineResult]:
        results = []
        for file_path in sorted(Path(dir_path).glob(pattern)):
            results.append(self.process_file(str(file_path)))
        return results
