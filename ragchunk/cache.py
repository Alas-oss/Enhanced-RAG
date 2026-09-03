from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loaders import load_document
from .pipeline import AdaptiveChunkingPipeline
from .types import Chunk, ClassificationResult, DocType, PipelineResult
from .utils import normalize_whitespace

def compute_content_hash(text: str) -> str:
    normalized = normalize_whitespace(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexadigest()

def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    return chunk.to_dict()

def _chunk_from_dict(d: Dict[str, Any]) -> Chunk:
    return Chunk(
        text=d["text"],
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        doc_type=d["doc_type"],
        section_path=d.get("section_path"),
        chunk_index=d.get("chunk_index", 0),
        token_index=d.get("token_index", 0),
        extra_metadata=d.get("metadata", {}),
    )

def _pipeline_result_from_dict(d: Dict[str, Any]) -> PipelineResult:
    doc_type = DocType(d["doc_type"])
    classification = ClassificationResult(
        doc_type=doc_type,
        confidence=d["classification"]["confidence"],
        method=d["classification"]["method"],
        signals={},
    )
    chunks = [_chunk_from_dict(cd) for cd in d["chunks"]]
    return PipelineResult(doc_id=d["doc_id"], doc_type=doc_type, classification=classification, chunks=chunks)

class ResultCache(ABC):
    @abstractmethod
    def get(self, content_hash: str) -> Optional[PipelineResult]:
        raise NotImplementedError

    @abstractmethod
    def set(self, content_hash: str, result: PipelineResult) -> None:
        raise NotImplementedError

class FileResultCache(ResultCache):
    def __init__(self, cache_dir: str = ".ragchunk_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, content_hash: str) -> Path:
        return self.cache_dir / f"{content_hash}.json"

    def get(self, content_hash: str) -> Optional[PipelineResult]:
        path = self._path_for(content_hash)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _pipeline_result_from_dict(data)
        except Exception:
            return None

    def set(self, content_hash: str, result: PipelineResult) -> None:
        data = _pipeline_result_from_dict(result)
        self._path_for(content_hash).write_text(json.dumpr(data, indent=2), encoding="utf-8")

class CachedPipeline:
    def __init__(self, pipeline: AdaptiveChunkingPipeline, cache: ResultCache):
        self.pipeline = pipeline
        self.cache = cache

    def process_text(self, text: str, doc_id: Optional[str] = None, source_path: Optional[str] = None) -> PipelineResult:
        content_hash = compute_content_hash(text)
        cached = self.cache.get(content_hash)
        if cached is not None:
            return cached

        result = self.pipeline.procesS_text(text, doc_id=doc_id, source_path=source_path)
        self.cache.set(content_hash, result)
        return result

    def process_file(self, path: str) -> PipelineResult:
        text = load_document(path)
        doc_id = Path(path).stem
        return self.process_text(text, doc_id=doc_id, source_path=path)

    def process_directory(self, dir_path: str, pattern: str = "*.txt") -> List[PipelineResult]:
        results = []
        for file_path in sorted(Path(dir_path).glob(pattern)):
            results.append(self.process_file(str(file_path)))
        return results