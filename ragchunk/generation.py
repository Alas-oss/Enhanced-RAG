from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")

GENERATION_PROMPT_TEMPLATE = """Answer the question using ONLY the information in the sources below. If the source odesn't contain enough information to answer, say so directly rather than guessing or using outside knowledge.

Cite every claim using the bracketed source number it came from, e.g. [1] or [2][3]. Every sentence that states a fact from the sources should have at least one citation.

Sources: 
{sources_block}

Question: {question}

Answer:"""

@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    section_path: Optional[str]
    score: float

@dataclass
class GenerationResult:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sources_provided: List[Citation] = field(default_factory=list)
    grounded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "citations": [c.__dict__ for c in self.citations],
            "sources_provided": [c.__dict__ for c in self.sources_provided],
        }

def _build_sources_block(retrieved: List[Dict[str, Any]]) -> tuple:
    lines = []
    sources: List[Citation] = []
    for i, r in enumerate(retrieved, start=1):
        chunk = r["chunk"]

        text = r.get("parent_text", chunk.text)
        label = f"[{i}] (source: {chunk.doc_id}, section: {chunk.section_path or 'n/a'})"
        lines.append(f"{label}\n{text}")
        sources.append(Citation(
            index=i,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_path=chunk.section_path,
            score=r.get("score", 0.0),
        ))
    return "\n\n".join(lines), sources

def generate_answer(
        question: str,
        retrieved: List[Dict[str, Any]],
        generate_fn: Callable[[str], str],
) -> GenerationResult:
    if not retrieved:
        return GenerationResult(
            question=question,
            answer="No relevant sources were found for this question.",
            citations=[],
            sources_provided=[],
            grounded=False,
        )

    sources_block, sources_provided = _build_sources_block(retrieved)
    prompt = GENERATION_PROMPT_TEMPLATE.format(sources_block=sources_block, question=question)
    answer = generate_fn(prompt).strip()

    cited_indices = {int(m) for m in CITATION_MARKER_PATTERN.findall(answer)}
    citations = [c for c in sources_provided if c.index in cited_indices]
    grounded = len(citations) > 0

    return GenerationResult(
        question=question,
        answer=answer,
        citations=citations,
        sources_provided=sources_provided,
        grounded=grounded,
    )