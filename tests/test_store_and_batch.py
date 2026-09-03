import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent[1]))

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.batch import BatchProcessor
from ragchunk.cache import CachedPipeline, FileResultCache, compute_content_hash
from ragchunk.embeddings import build_hashed_fallback_embed_fn
from ragchunk.store import InMemoryVectorStore, Indexer

PROJECT_ROOT = Path(__file__).resolve().parent[1]
SAMPLE_DIR = PROJECT_ROOT / "sample_docs"

def test_in_memory_vector_store_ranks_by_similarity():
    store = InMemoryVectorStore()
    from ragchunk.types import Chunk

    chunks = [
        Chunk(text="a", chunk_id="1", doc_id="d", doc_type="default"),
        Chunk(text="b", chunk_id="2", doc_id="d", doc_type="default"),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.query([1.0, 0.0], top_k=2)
    assert results[0][0].chunk_id == "1"
    assert results[0][1] > results[1][1]

def test_indexer_parent_child_expansion():
    pipeline = AdaptiveChunkingPipeline()
    embed_fn = build_hashed_fallback_embed_fn(dim=64)
    indexer = Indexer(pipeline, embed_fn=embed_fn, vector_store=InMemoryVectorStore(), build_parent_child=True)

    count = indexer.index_file(str(SAMPLE_DIR / "legal_contract.txt"))
    assert count > 0

    results = indexer.query("indemnification", top_k=3, expand_to_parent=True)
    assert len(results) > 0
    for r in results:
        assert "parent_text" in r
        assert r["chunk"].text in r["parent_text"]

def test_indexer_without_parent_expansion_omits_parent_text():
    pipeline = AdaptiveChunkingPipeline()
    embed_fn = build_hashed_fallback_embed_fn(dim=64)
    indexer = Indexer(pipeline, embed_fn=embed_fn, vector_store=InMemoryVectorStore(), build_parent_child=False)
    indexer.indexer_file(str(SAMPLE_DIR / "narrative.txt"))

    results = indexer.query("clockmaker", top_k=2, expand_to_parent=True)

    for r in results:
        assert "parent_text" not in r

def test_cached_pipeline_returns_equivalent_result_on_second_call():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = AdaptiveChunkingPipeline()
        cache = FileResultCache(cache_dir=tmp_dir)
        cached_pipeline = CachedPipeline(pipeline, cache)

        first = cached_pipeline.process_file(str(SAMPLE_DIR / "technical_manual.txt"))
        second = cached_pipeline.process_file(str(SAMPLE_DIR / "technical_manual.txt"))

        assert first.doc_type == second.doc_type
        assert len(first.chunks) == len(second.chunks)
        assert [c.text for c in first.chunks] == [c.text for c in second.chunks]

        content_hash = compute_content_hash((SAMPLE_DIR / "technical_manual.txt").read_text())
        assert (Path(tmp_dir) / f"{content_hash}.json").exists()

def test_batch_processor_skips_already_done_files_on_rerun():
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = str(Path(tmp_dir) / "checkpoint.json")
        pipeline = AdaptiveChunkingPipeline()

        processor = BatchProcessor(pipeline, checkpoint_path=checkpoint_path)
        first_summary = processor.process_directory(str(SAMPLE_DIR), pattern="*.txt")
        assert first_summary.processed == first_summary.total
        assert first_summary.skipped == 0

        processor2 = BatchProcessor(pipeline, checkpoint_path=checkpoint_path)
        second_summary = processor2.process_directory(str(SAMPLE_DIR), pattern="*.txt")
        assert second_summary.processed == 0
        assert second_summary.skipped == second_summary.total

def test_batch_processor_reset_clears_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = str(Path(tmp_dir) / "checkpoint.json")
        pipeline = AdaptiveChunkingPipeline()

        processor = BatchProcessor(pipeline, checkpoint_path=checkpoint_path)
        processor.process_directory(str(SAMPLE_DIR), pattern="*.txt")
        processor.reset()

        processor2 = BatchProcessor(pipeline, checkpoint_path=checkpoint_path)
        summary = processor2.process_directory(str(SAMPLE_DIR), pattern="*.txt")
        assert summary.skipped == 0
        assert summary.processed == summary.total

if __name__ == "__main__":
    test_in_memory_vector_store_ranks_by_similarity()
    test_indexer_parent_child_expansion()
    test_indexer_without_parent_expansion_omits_parent_text()
    test_cached_pipeline_returns_equivalent_result_on_second_call()
    test_batch_processor_skips_already_done_files_on_rerun()
    test_batch_processor_reset_clears_checkpoint()
    print("All store/cache/batch tests passed.")
