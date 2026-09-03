from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..pipeline import AdaptiveChunkingPipeline
from ..types import Chunk
from .base import VectorStore
from .in_memory import InMemoryVectorStore

EmbedFn = Any

class Indexer:
    def __init__(
        self,
        pipeline: AdaptiveChunkingPipeline,
        embed_fn: EmbedFn,
        vector_store: Optional[VectorStore] = None,
        build_parent_child: bool = True,
    ):
        self.pipeline = pipeline
        self.embed_fn = embed_fn
        self.vector_store = vector_store or InMemoryVectorStore()
        self.build_parent_child = build_parent_child

        self._parent_texts: Dict[str, str] = {}
        self._child_to_parent: Dict[str, str] = {}

    def index_file(self, path: str) -> int:
        result = self.pipeline.process_file(path)
        return self.index_chunks(result.chunks)

    def index_directory(self, dir_path: str, pattern: str = "*.txt") -> int:
        total = 0
        for file_path in sorted(Path(dir_path).glob(pattern)):
            total += self.index_file(str(file_path))
        return total
    def index_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        if self.build_parent_child:
            self._register_parent_child(chunks)

        texts = [c.text for c in chunks]
        embeddings = self.embed_fn(texts)
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    def query(self, text: str, top_k: int = 5, expand_to_parents: bool = True) -> List[Dict[str, Any]]:
        query_embedding = self.embed_fn([text])[0]
        raw_results = self.vector_store.query(query_embedding, top_k=top_k)

        output = []
        for chunk, score in raw_results:
            entry: Dict[str, Any] = {"chunk": chunk, "score": score}
            if expand_to_parents and self.build_parent_child:
                parent_id = self._child_to_parent.get(chunk.chunk_id)
                entry["parent_id"] = parent_id
                entry["parents_text"] = self._parent_texts.get(parent_id, chunk.text)
            output.append(entry)
        return output

    def _register_parent_child(self, chunks: List[Chunk]) -> None:
        groups: Dict[tuple[str, Optional[str]], List[Chunk]] = {}
        for c in chunks:
            key = (c.doc_id, c.section_path)
            groups.setdefault(key, []).append(c)

        for (doc_id, section_path), group_chunks in groups.items():
            parent_id = f"{doc_id}::{section_path or 'root'}"
            ordered = sorted(group_chunks, key=lambda c: c.chunk_index)
            self._parent_texts[parent_id] = "\n\n".join(c.text for c in ordered)
            for c in ordered:
                self._child_to_parent[c.chunk_id] = parent_id