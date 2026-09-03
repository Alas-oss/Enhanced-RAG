from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .classifier import HybridClassifier
from .loaders import load_document
from .router import ChunkerRouter
from .section_splitter import split_into_sections
from .types import ClassificationResult, Document, DocType, PipelineResult
from .utils import normalize_whitespace


class AdaptiveChunkingPipeline:
    def __init__(
        self,
        llm_classify_fn: Optional[Callable[[str], str]] = None,
        router: Optional[ChunkerRouter] = None,
        narrative_embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.classifier = HybridClassifier(llm_classify_fn=llm_classify_fn)
        self.router = router or ChunkerRouter(narrative_embed_fn=narrative_embed_fn)

    def process_text(self, text: str, doc_id: Optional[str] = None, source_path: Optional[str] = None) -> PipelineResult:
        doc_id = doc_id or str(uuid.uuid4())
        clean_text = normalize_whitespace(text)

        raw_sections = split_into_sections(clean_text)
        
        if len(raw_sections) <= 1:
            return self._classify_and_chunk_single(clean_text, doc_id, source_path)

        section_results = [
            (label, sec_text, self.classifier.classify(sec_text))
            for label, sec_text in raw_sections
        ]
        groups = self._merge_consecutive_by_type(section_results)

        if len(groups) == 1:
            _, group_text, group_classification = groups[0]
            return self._chunk_with_classificaiton(group_text, group_classification, doc_id, source_path)

        return self._chunk_mixed_groups(groups, doc_id, source_path)

    def process_file(self, path: str) -> PipelineResult:
        text = load_document(path)
        doc_id = Path(path).stem
        return self.process_text(text, doc_id=doc_id, source_path=path)

    def process_directory(self, dir_path: str, pattern: str = "*.txt") -> List[PipelineResult]:
        results = []
        for file_path in sorted(Path(dir_path).glob(pattern)):
            results.append(self.process_file(str(file_path)))
        return results

    def _classify_and_chunk_single(self, text: str, doc_id: str, source_path: Optional[str]) -> PipelineResult:
        classification = self.classifier.classify(text)
        return self._chunk_with_classification(text, classification, doc_id, source_path)

    def _chunk_with_classificaiton(
            self, text: str, classification: ClassificationResult, doc_id: str, source_path: Optional[str]
    ) -> PipelineResult:
        document = Document(doc_id=doc_id, text=text, source_path=source_path)
        chunker = self.router.get_chunker(classification.doc_type)
        chunks = chunker.chunk(document)
        return PipelineResult(
            doc_id=doc_id,
            doc_type=classification.doc_type,
            classification=classification,
            chunks=chunks,
        )

    def _merge_consecutive_by_type(
            self, sections_results: List[Tuple[Optional[str], str, ClassificationResult]]
    ) -> List[Tuple[Optional[str], str, ClassificationResult]]:
        groups: List[Tuple[Optional[str], str, ClassificationResult]] = []
        for label, text, classification in sections_results:
            if groups and groups[-1][2].doc_type == classification.doc_type:
                prev_label, prev_text, prev_classification = groups[-1]
                groups[-1] = (prev_label, prev_text + text, prev_classification)
            else:
                groups.append((label, text, classification))
        return groups

    def _chunk_mixed_groups(
        self, 
        groups: List[Tuple[Optional[str], str, ClassificationResult]],
        doc_id: str, 
        source_path: Optional[str],
    ) -> PipelineResult:
        all_chunks = []
        chunk_offset = 0
        for group_label, group_text, group_classification in groups:
            sub_doc = Document(doc_id=doc_id, text=group_text, source_path=source_path)
            chunker = self.router.get_chunker(group_classification.doc_type)
            sub_chunks = chunker.chunk(sub_doc)
            for c in sub_chunks:
                c.chunk_index += chunk_offset
                c.chunk_id = f"{doc_id}::chunk_{c.chunk_index}"
                c.extra_metadata["detect_section_label"] = group_label
            chunk_offset += len(sub_chunks)
            all_chunks.extend(sub_chunks)

        overall_type = self._majority_doc_type(groups)
        overall_classification = ClassificationResult(
            doc_type=overall_type,
            confidence=1.0,
            method="section_level_mixed",
            signals={
                "section_count": len(groups),
                "section_types": [g[2].doc_type.value for g in groups],
            },
        )
        return PipelineResult(
            doc_id=doc_id,
            doc_type=overall_type,
            classification=overall_classification,
            chunks=all_chunks,
        )

    @staticmethod
    def _majority_doc_type(groups: List[Tuple[Optional[str], str, ClassificationResult]]) -> DocType:
        counts: Counter = Counter()
        for _, text, classification in groups:
            counts[classification.doc_type] += len(text)
        return counts.most_common(1)[0][0]