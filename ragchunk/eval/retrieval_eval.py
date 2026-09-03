from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..pipeline import AdaptiveChunkingPipeline
from ..types import Chunk
from .dataset import QAExample
from .retriever import TfidfRetriever

@dataclass 
class RetrievalEvalReport:
    total: int
    hits: int
    hit_rate: float
    mean_reciprocal_rank: float
    per_type_hit_rate: Dict[str, float]
    details: List[Dict[str, Any]]

def run_retrieval_eval(
        qa_examples: List[QAExample],
        pipeline: Optional[AdaptiveChunkingPipeline] = None,
        base_dir: Optional[str] = None,
) -> RetrievalEvalReport:
    pipeline = pipeline or AdaptiveChunkingPipeline()
    chunk_cache: Dict[str, List[Chunk]] = {}

    details: List[Dict[str, Any]] = []
    reciprocal_ranks: List[float] = []
    per_type_hits: Dict[str, List[int]] = {}

    for qa in qa_examples:
        doc_path = str(Path(base_dir) / qa.path) if base_dir else qa.path
        if doc_path not in chunk_cache:
            result = pipeline.process_file(doc_path)
            chunk_cache[doc_path] = result.chunks

        chunks = chunk_cache[doc_path]
        if not chunks:
            continue

        retriever = TfidfRetriever([c.text for c in chunks])
        ranked = retriever.rank(qa.question)

        expected_indices = [
            i for i, c in enumerate(chunks)
            if c.section_path and qa.expected_section_contains.lower() in c.section_path.lower()
        ]

        top_k_indices = [idx for idx, _ in ranked[: qa.top_k]]
        hit = any(idx in expected_indices for idx in top_k_indices)
        rank_of_first_expected = None
        for rank_pos, (idx, _) in enumerate(ranked, start=1):
            if idx in expected_indices:
                rank_of_first_expected = rank_pos
                break
        reciprocal_rank = 1.0 / rank_of_first_expected if rank_of_first_expected else 0.0
        reciprocal_ranks.append(reciprocal_rank)

        per_type_hits.setdefault(qa.doc_type, []).append(1 if hit else 0)

        details.append({
            "path": qa.path,
            "question": qa.question,
            "expected_section_contains": qa.expected_section_contains,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "top_result_section": chunks[ranked[0][0]].section_path if ranked else None,
        })

    total = len(details)
    hits = sum(1 for d in details if d["hit"])
    hit_rate = hits / total if total else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    per_type_hit_rate = {t: sum(v) / len(v) for t, v in per_type_hits.items()}

    return RetrievalEvalReport(
        total=total,
        hits=hits,
        hit_rate=hit_rate,
        mean_reciprocal_rank=mrr,
        per_type_hit_rate=per_type_hit_rate,
        details=details,
    )