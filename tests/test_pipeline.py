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
    
    for chunk in result.chunks:
        assert chunk.token_estimate <= 300  


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
    assert len(results) == 6
    for result in results:
        for chunk in result.chunks:
            assert chunk.chunk_id
            assert chunk.doc_id == result.doc_id
            assert chunk.token_estimate > 0
            assert chunk.text.strip() != ""


def test_uniform_documents_unaffected_by_section_level_classification():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "legal_contract.txt"))

    assert result.doc_type == DocType.LEGAL_CONTRACT
    assert result.classification.method == "heuristic"


def test_financial_report_extracts_tables_atomically():
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "financial_report.txt"))

    assert result.doc_type == DocType.FINANCIAL_REPORT
    table_chunks = [c for c in result.chunks if c.extra_metadata.get("content_type") == "table"]
    assert len(table_chunks) >= 2  # two tables in the sample doc
    for c in table_chunks:
        assert c.text.strip().startswith("|")


def test_mixed_document_produces_multiple_doc_types():
    """The sample doc mixes a government regulation section with a
    technical API appendix -- section-level classification should detect
    both types within the same PipelineResult."""
    pipeline = AdaptiveChunkingPipeline()
    result = pipeline.process_file(str(SAMPLE_DIR / "mixed_gov_technical.txt"))

    chunk_types = {c.doc_type for c in result.chunks}
    assert len(chunk_types) >= 2
    assert "government_regulation" in chunk_types
    assert "technical_manual" in chunk_types
    assert result.classification.method == "section_level_mixed"


if __name__ == "__main__":
    test_legal_contract_chunks_have_section_paths()
    test_government_reg_chunks_are_small_and_precise()
    test_technical_manual_keeps_code_blocks_atomic()
    test_narrative_chunks_respect_chapter_boundaries()
    test_all_chunks_have_required_fields()
    test_uniform_documents_unaffected_by_section_level_classification()
    test_financial_report_extracts_tables_atomically()
    test_mixed_document_produces_multiple_doc_types()
    print("All pipeline tests passed.")
