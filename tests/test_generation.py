import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.embeddings import build_hashed_fallback_embed_fn
from ragchunk.generation import generate_answer
from ragchunk.store.in_memory import InMemoryVectorStore
from ragchunk.store.indexer import Indexer 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "sample_docs"

def _fake_retrieved(n=2):
    from ragchunk.types import Chunk

    retrieved = []
    for i in range(1, n + 1):
        chunk = Chunk(
            text=f"This is source text number {i}.",
            chunk_id=f"chunk_{i}",
            doc_id="doc_a",
            doc_type="legal_contract",
            section_path=f"Section {i}",
        )
        retrieved.append({"chunk": chunk, "score": 0.9 - 0.1 * i})
    return retrieved

def test_no_sources_returns_safe_message_without_calling_the_model():
    def explode(prompt: str) -> str:
        raise AssertionError("generate_fn should never be called with zero retrieved sources")

    result = generate_answer("any question", [], explode)

    assert result.grounded is False
    assert result.citations == []
    assert "No relevant sources" in result.answer

def test_prompt_includes_numbered_sources_and_metadata():
    captured_prompt = {}

    def capture_fn(prompt: str) -> str:
        captured_prompt["value"] = prompt
        return "An answer with no citations"

    generate_answer("a question", _fake_retrieved(2), capture_fn)

    prompt = captured_prompt["value"]
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "doc_a" in prompt
    assert "Section 1" in prompt
    assert "Section 2" in prompt

def test_citations_are_parsed_from_the_answer_text():
    def fn(prompt: str) -> str:
        return "The first source says X [1]. The second source is unrelated and not user."

    result = generate_answer("a question", _fake_retrieved(2), fn)

    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].chunk_id == "chunk_1"
    assert len(result.sources_provided) == 2

def test_answer_without_citation_markers_is_not_grounded():
    def fn(prompt: str) -> str:
        return "This answer has no bracketed citations at all."

    result = generate_answer("a question", _fake_retrieved(2), fn)

    assert result.grounded is False
    assert result.citations == []
    assert len(result.sources_provided) == 2

def test_end_to_end_index_query_generate_with_fake_llm():
    def fake_generate_fn(prompt: str) -> str:
        return "Section 3.1 covers indemnification for patent infringement claims [1]."

    pipeline = AdaptiveChunkingPipeline()
    embed_fn = build_hashed_fallback_embed_fn(dim=64)
    indexer = Indexer(pipeline, embed_fn=embed_fn, vector_store=InMemoryVectorStore())
    indexer.index_file(str(SAMPLE_DIR / "legal_contract.txt"))

    retrieved = indexer.query("indemnification patent infringement", top_k=3)
    result = generate_answer("What does indemnificaiton cover?", retrieved, fake_generate_fn)

    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == retrieved[0]["chunk"].chunk_id
    assert result.citations[0].doc_id == "legal_contract"

if __name__ == "__main__":
    test_no_sources_returns_safe_message_without_calling_the_model()
    test_prompt_includes_numbered_sources_and_metadata()
    test_citations_are_parsed_from_the_answer_text()
    test_answer_without_citation_markers_is_not_grounded()
    test_end_to_end_index_query_generate_with_fake_llm()
    print("All generation tests passed.")