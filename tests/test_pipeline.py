import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchunk import AdaptiveChunkingPipeline, DocType

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"


def test_legal_contract_chunks_have_section_paths():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "legal_contract.txt"))

    assert result.doc_type == DocType.LEGAL_CONTRACT
    assert len(result.chunks) > 1
    labeled = [c for c in result.chunks if c.section_path]
    assert len(labeled) > 0
    assert any("Article" in (c.section_path or "") or "Section" in (c.section_path or "") for c in result.chunks)


def test_government_reg_chunks_are_small_and_precise():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "government_regulation.txt"))

    assert result.doc_type == DocType.GOVERNMENT_REGULATION
    # Gov reg chunker uses a tighter token budget (250) than default (500)
    for chunk in result.chunks:
        assert chunk.token_estimate <= 300  # small overrun allowed for atomic sections


def test_technical_manual_keeps_code_blocks_atomic():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "technical_manual.txt"))

    code_chunks = [c for c in result.chunks if c.extra_metadata.get("content_type") == "code_block"]
    assert len(code_chunks) > 0
    for c in code_chunks:
        assert c.text.strip().startswith("```")


def test_narrative_chunks_respect_chapter_boundaries():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "narrative.txt"))

    assert result.doc_type == DocType.NARRATIVE_PROSE
    section_paths = {c.section_path for c in result.chunks}
    assert any(sp and "Chapter" in sp for sp in section_paths)


def test_all_chunks_have_required_fields():
    pipeline = AdaptiveChunkingPipeline()
    results = pipeline.process_directory(str(SAMPLE_DIR), pattern="*.txt")
    assert len(results) == 4
    for result in results:
        for chunk in result.chunks:
            assert chunk.chunk_id
            assert chunk.doc_id == result.doc_id
            assert chunk.token_estimate > 0
            assert chunk.text.strip() != ""


if __name__ == "__main__":
    test_legal_contract_chunks_have_section_paths()
    test_government_reg_chunks_are_small_and_precise()
    test_technical_manual_keeps_code_blocks_atomic()
    test_narrative_chunks_respect_chapter_boundaries()
    test_all_chunks_have_required_fields()
    print("All pipeline tests passed.")
