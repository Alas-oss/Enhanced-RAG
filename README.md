# Adaptive RAG Chunking Pipeline

A classification-driven chunking pipeline for retrieval-augmented generation. Every document is inspected before it's split, so the pipeline works out what *kind* of document it's looking at, then routes it to a chunking strategy built for that content, instead of applying one fixed chunk size to everything that comes through.

## Purpose of this project

Most RAG pipelines chunk every document the same way, a fixed token count with some overlap 0 regardless of what the document actually is. That works poorly the moment the corpus isn't uniform:

- **A legal contract split mid-clause loses its meaning.** 
- **A government regulation needs precision, not breadth.**
- **A code sample split in two is entirely useless.**
- **A financial table split across chunk boundaries loses its functionality of a data-carrying table.**
- **In standard practice, nobody will be routing per-document or per-section.**

This pipeline addresses this with one classify-then-route flow, rather than hand-picking a chunking strategy per source file.

## Architecture

```
 document (.txt / .md / .pdf / .docx)
      │
      ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                 Adaptive Chunking Pipeline                    │
 │                                                               │
 │  1. Load & normalize              (ragchunk/loaders.py)       │
 │  2. Coarse section pre-split       (ragchunk/section_splitter)│
 │  3. Per-section classification      (ragchunk/classifier.py)  │
 │       heuristic scoring first, LLM fallback only if ambiguous │
 │  4. Same-type section merging        (ragchunk/pipeline.py)   │
 │  5. Strategy routing                   (ragchunk/router.py)   │
 │  6. Type-specific chunking              (ragchunk/chunkers/)  │
 │  7. Metadata enrichment & packaging      (ragchunk/types.py)  │
 └───────────────────────────────────────────────────────────────┘
      │
      ▼
 Embeddable Chunk objects, ready for a vector store
```
Every stage is independently testable; `main.py` wires them together into one CLI command that accepts a file or a directory.

## Features

### Hybrid classification

Every document it scored against a set of regex/keyword heuristics for each known type (`ragchunk/classifier.py`) - which is fast, free, deterministic and does not require a network call. A document only escalates to an LLM classification call when the heuristic result is genuinely ambiguous: low confidence, or too close a race between the top two candidate types. This keeps LLM spend proportional to the hard cases instead of paying for a classification call on every document. The LLM step is adjustable, you can pass any `Callable [[str], str]` into `HybridClassifier`, without touching classification logic itself.

### Section-level, not just document-level, classification

A document is coarsely pre-split on structural boundaries (Article / Title / Part / Chapter / Appendix / Heading) before classification runs (`ragchunk/section_splitter.py`). Each section is classified independently, and consecutive sections of the same type are merged back together. A uniform document collapses back into a single classification pass automatically, so the section-level logic only changes behavior for documents that genuinely mix types, like a regulation with a technical API appendix attached, which now produces multiple correctly-typed chunk groups within one result instead of forcing the whole file through one strategy.

### Type-specific chunking strategies

Each document type routes to a dedicated chunker (`ragchunk/chunkers/`), all sharing common bookkeeping via a base class but implementing their own splitting rules:

- **Legal contracts** - split on Article/Section/WHEREAS boundaries; every chunk is tagged with its clause label so a retrieved answer traces back to it
- **Government/regulatory text** - hierarchical Title → Part → Subpart → § splitting with a tighter token budget (precision over breadth); chunks are flagged when they contain a statutory citation
- **Technical manuals** - split on markdown headings; code blocks are kept as atomic units regardless of token budget, since a broken code sample is worse than an oversized chunk
- **Financial reports** - tables are detected and extracted as atomic chunks (never split mid-table); narrative text around them is chunked normally
- **Narrative prose** - paragraph-packing that respects chapter boundaries as hard breaks; optionally supports embedding-breakpoint chunking (split where meaning actually shifts between paragraphs) via a pluggable `embed_fn`, falling back to plain packing when none is provided
- **Default** - recursive paragraph → sentence splitting, used for anything unclassified or without a dedicated strategy

### Multi-format ingestion

`ragchunk/loaders.py` reads `.txt`, `.md`, `.pdf`, and `.docx` directly, dispatched by file extension. PDF text is extracted page-by-page; DOCX ingestion also pulls table content out of Word tables (not just paragraph text) so it reaches the financial chunker's table detector. PDF/DOCX support are optional dependencies, so if `pypdf` or `python-docx` aren't installed, reading that file type raises a clear, actionable error instead of a stack trace; `.txt`/`.md` files always work with zero dependencies.

### Real token accounting

Chunk sizes are budgeted against actual token counts via `tiktoken`, not a word-count guess, so `max_tokens` settings per chunker translate to real retrieval-relevant chunk sizes. Falls back automatically to a word-count approximation if `tiktoken` isn't installed, so the pipeline never hard-requires it.

### Full provenance metadata

Every chunk carries `doc_type`, `section_path` (e.g. a full Title/Path/Subpart trail), `chunk_index`, `token_estimate`, and a flexible `extra_metadata` dict(citation flags, code-block/table markers, which coarse section a mixed document's chunk came from). This is what makes a retrieved chunk traceable back to exactly where it came from, rather than an anonymous chunk of text.

### Evaluation harness

`run_eval.py` runs the pipeline against two labeled datasets in `eval_data/` and reports three things: **classifier accuracy** (precision/recall/F1 per document type, run through the full pipeline rather than the classifier in isolation, so section-level classification bugs get caught too), **retrieval quality** (a dependency-free TF-IDF retriever check whether the chunk actually containing the answer to a sample question shows up in the top-k results, and this catches a different failure mode than classification correctness: a document can be typed correctly and still chunked in a way that separates a question from its answer), and **threshold calibration** (sweeps `HybridClassifier`'s confidence/margin thresholds and recommends the most cost-efficient setting that doesn't sacrifice accuracy on the labeled set). All three write to a single JSON report via `--output`.

### Vector store integration and parent-child retrieval

`ragchunk/store/` embeds chunks via a pluggable `embed_fn` (the same `Callable[[List[str]], List[List[float]]]` pattern used by the narrative chunker's semantic mode) into a pluggable `VectorStore` - `InMemoryVectorStore` by default (zero dependencies), or `ChromaVectorStore` for persistence across runs. `Indexer` also implements parent-child retrieval: every chunk sharing the same `(doc_id, section_path)` is reassembled into ` "parent", therefore the full clause, §. heading, or chapter it came from, with no changes needed to any chunker, since this falls directly out of metadata they already produce. A query returns the precise, small chunk that matched (good for retrieval precision) alongside its full parent text (good for generation context), letting you retrieve narrow and generate broad.

### Caching and batch processing

`ragchunk/cache.py` adds a content-hash-keyed cache (`CachedPipeline`) that wraps `AdaptiveChunkingPipeline` transparently, re-running the pipeline on unchanged document content skips classification and chunking entirely and returns the cached result, keyed by content rather than filename, so a renamed file with identical content still hits the cache. `ragchunk/batch.py` adds `BatchProcessor`, which processes a whole directory with resumablecheckpoint: prgress is saved after every single file, so an interrupsted batch job over a large corpus resumes from where it stopped instead or restarting. The two compose naturally using `run_batch.py --cache-dir` gets you both file-level checkpoint skipping and content-level cache hits in one run.

## Project strcture

```
ragchunk/
  types.py            Shared data structures (Chunk, Document, DocType, ...)
  utils.py             Token estimation (tiktoken, with fallback), text cleanup
  loaders.py            Multi-format document ingestion (.txt/.md/.pdf/.docx)
  classifier.py          Hybrid heuristic + pluggable-LLM classification
  section_splitter.py     Coarse structural pre-split for mixed-type detection
  router.py                 Maps a classified doc type to its chunker
  pipeline.py                 Orchestrates load -> classify -> route -> chunk
  chunkers/
    base.py                    Shared scaffolding every chunker builds on
    legal.py                    Legal contract chunking rules
    government.py                Government/regulatory chunking rules
    technical.py                  Technical manual chunking rules
    financial.py                   Financial report / table-aware chunking rules
    narrative.py                    Prose chunking, optional semantic mode
    default.py                       Fallback for anything else
  eval/
    dataset.py               Labeled dataset + QA dataset loaders
    metrics.py                 Precision/recall/F1, confusion matrix (no sklearn)
    classifier_eval.py           Runs the labeled set through the full pipeline
    retriever.py                   Dependency-free TF-IDF retriever, eval-only
    retrieval_eval.py                Checks whether the right chunk is retrievable
    calibration.py                     Sweeps classifier thresholds
    report.py                            Console + JSON reporting
  store/
    base.py                 Abstract VectorStore interface
    in_memory.py              Zero-dependency default vector store
    chroma_store.py             Optional Chroma-backed persistent store
    indexer.py                    Embeds chunks, handles parent-child expansion
  cache.py                Content-hash result caching (CachedPipeline)
  batch.py                 Resumable checkpointed batch processing
  embeddings.py             Reference embed_fn implementations (Gemini + local fallback)
eval_data/               Labeled dataset and QA dataset (plain JSON)
sample_docs/            One example per document type, including a mixed-type sample
tests/                  Classifier, pipeline, utility, eval, and store/batch test suites
main.py                 Command-line entry point
run_eval.py              Evaluation harness entry point
run_index.py              Indexing & retrieval CLI (vector store + parent-child)
run_batch.py               Batch processing CLI with checkpointing
requirements.txt        All dependencies are optional; see below
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The core pipeline runs with **zero required dependencies** - heuristic classification and every chunking strategy is pure standard library. Everything in `requirements.txt` is optional and the pipeline degrades gracefully without it:

| Package | Enables | Without it |
|---|---|---|
| `tiktoken` | Exact token counts | Falls back to a word-count approximation | 
| `pypdf` | Reading `.pdf` files | Reading a PDF raises a clear `ImportError` telling you to install it |
| `python-docx` | Reading `.docx` files | Same graceful error as above, for Word files |
| `google-genai` (or any other similar API key provider) | LLM fallback classification for ambiguous documents, and real semantic embeddings for `--embed` / `run_index.py` | Ambiguous documents are still classified via the heuristics, falling back to plain paragraph packing; `run_index.py` falls back to a local hashed-vector stand-in (not to semantic search) |
| `chromadb` | Persisted vector storage across runs (`ChunkVectorStore`) | `InMemoryVectorStore` (the default) works with zero dependencies, but doesn't persist between process runs |

### Configuration

The LLM classification fallback is provider-agnostic by design, therefore both `HybridClassifier` and `AdaptiveChunkingPipeline` accept any `llm_classify_fn: Callable[[str], str]`. `main.py`'s `build_llm_classify_fn()` shows a working example wired to Google AI Studio (Gemini); swap the client construction there for others ir swaping to a different provider. Set the API key as an environment variable before running with `--use-llm`:

```bash
export GOOGLE_API_KEY="..."
```

## Running the pipeline

```bash 
# a single file (and of .txt / .md / .pdf / .docx)
python main.py sample_docs/legal_contract.txt

# a whole directory, with results written to JSON
python main.py sample_docs/ --dir --output results.json

# enable the LLM fallback for ambiguous documents
python main.py sample_docs/legal_contract.txt --use-llm
```

Programmatic use:
```python
from ragchunk import AdaptiveChunkingPipeline

pipeline = AdaptiveChunkingPipeline()
result = pipeline.process_file("my_document.pdf")

print(result.doc_type, result.classification.method)
for chunk in result.chunks: 
    print(chunk.section_path, chunk.token_estimate, chunk.text[:60])
```

## Running the tests

```bash
python tests/test_classifier.py
python tests/test_pipeline.py
python tests/test_utils.py
python tests/test_eval.py
pytest tests/
```

## Running the evaluation harness

```bash
python run_eval.py
python run_eval.py --output eval_report.json
python run_eval.py --skip-retrieval
```

## Indexing and querying

```bash
# index a directory and run a query, using the local fallback embedder
# (no API key is needed here, but this is NOT semantic search)
python run_index.py sample_docs/ --dir --query "What is the governing law?"

python run_index.py sample_docs/ --dir --store chroma --persist-dir ./chroma_db --query "..."
```

Each result includes both the precise matched chunk and, by default, its full parent section (`parent_text`) - pass `--no-parent-expansion` to retrieve narrow without the wider context.

## Batch processing large corpora

```bash
python run_batch.py large_corpus/ --pattern "*.pdf" --checkpoint batch_state.json
# interrupt it, then re-run the same command -- already-processed files are skipped
python run_batch.py large_corpus/ --pattern "*.pdf" --checkpoint batch_state.json

# combine with content-hash caching
python run_batch.py large_corpus/ --checkpoint batch_state.json --cache-dir .ragchunk_cache

# force a full process (e.g. after a chunking rule change)
python run_batch.py large_corpus/ --checkpoint batch_state.json --reset
```

---

## What's implemented

- Hybrid heuristic + pluggable-LLM document classification
- Section-level classification for documents that mix types, with automatic fallback to whole-document classification for uniform documents
- Six chunking strategies: legal, government/regulatory, technical, financial (table-aware), narrative (with optional semantic mode), and a generic fallback
- Multi-format ingestion: `.txt`, `.md`, `.pdf`, `.docx`
- Real token-budgeted chunk sizing via `tiktoken`, with graceful fallback
- Full chunk-level provenance metadata (section path, citation/table/code flags, token counts)
- An evaluation harness: classifier precision/recall/F1, retrieval-quality (hit-rate/MRR) checking, and threshold calibration, all against a growable labeled dataset pluggable vector store integration (in-memory by default, optional Chroma persistence) with parent-child retrieval indexing content-hash result caching and resumable, checkpointed batch processing for large document sets
- CLI wiring for semantic narrative chunking (`--embed`)
- Result caching (`--cache-dir`), indexing/querying (`run_index.py`)
- Batch processing (`run_batch.py`)