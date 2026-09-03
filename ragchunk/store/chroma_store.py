from __future__ import annotations

import json
from typing import List, Optional, Tuple

from ..types import Chunk 
from .base import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "ragchunk", persist_directory: Optional[str] = None):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires chromadb. Install with: pip install chromadb"
            ) from exc

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings -- must match one-to-one.")
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_type": c.doc_type,
                    "section_path": c.section_path or "",
                    "chunk_index": c.chunk_index,
                    "token_estimate": c.token_estimate,
                    "extra_metadata_json": json.dumps(c.extra_metadata),
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        results = self._collection.query(query_embedding=[query_embedding], n_results=top_k)
        if not results["ids"] or not results["ids"][0]:
            return []

        pairs = []
        for id_, doc_text, meta, distance in zip(
            results["ids"[0], results["documents"][0], results["metadatas"][0], results["distances"][0]]
        ):
            chunk = Chunk(
                text=doc_text,
                chunk_id=id_,
                doc_id=meta.get("doc_id", ""),
                doc_type=meta.get("doc_type", "default"),
                section_path=meta.get("section_path") or None,
                chunk_index=meta.get("chunk_index", 0),
                token_estimate=meta.get("token_estimate", 0),
                extra_metadata=json.loads(meta.get("extra_metadata_json", "{}")),
            )
            similarity = 1 - distance
            pairs.append((chunk, similarity))
        return pairs