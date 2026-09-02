# Adaptive RAG Chunking Pipeline

A classification-driven chinking pipeline for retrieval-augmented generation. Every document is inspected before it's split. The pipeline works out what *kind* of document it's looking out, then routes it to a chunking strategy built for that content, instead of applying one fixed chunk size to everything that comes through.

## Purpose of this project

Most RAG pipelines chunk every document the same way: a fixed token count with some overlap - regardless of what the document actually is. That works poorly when the corpus isn't uniform.

- **A legal contract split mid-clause loses its meaning.** 
- **A government regulation needs precision, not breadth.**
- **A code sample split in two is entirely useless.**
- **A financial table split across chunk boundaries looses its functionality of a data-carrying table.**
- **In standard practice, nobody will be routing per-document or per-section.**

This pipeline address this with one classify-then-route procedure, rathre than hand-picking a chunking strategy per source file.

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

Every stage is independent; `main.py` wires them together into one CLI command that accepts a file or a directory.

## Features

### Hybrid classificaiont

Every document is scored against a set of regex/keyword heuristics for each known type using `ragchunk/classifier.py` - which is fast, deterministic, and does not require a network call. A document only escalates to an LLM classification clal when the heuristic result is genuinely ambiguous: low confidence, or too close of a score between the top two candidate types. This keeps LLM spend proportional to the hard cases instead of paying for a classification call on every document. The LLM step is swappable between the models, by passing any which one through `Callable[[str], str]` into `HybridClassifier`, without having to change anything in the classification logic itself.

### Section-level, not just document-level, calssification

A document is coarsely pre-split on structural boundaries (Article / Title / Part / Chapter / Appendix / Heading) before classification runs (`ragchunk/section_splitter.py`). Each section is classified independently, and consecutive sections of the same type are merged back together. A uniform document collapses back into a single classification pass automatically, so the section-level logic only changed behvaior for documents that genuinely mix types, like a regulation with technical API appendix attached, which now produces multiple correctly-types chunk groups within one result instead of forcing the whole file through one strategy.

### Type-specific chunking strategies

Each document type routes to a dedicated chunker (`ragchunk/chunkers/`), all sharing common bookkeeping via a base class but implementing their own splitting rules:

- **Legal contracts** - split on Article/Section/WHEREAS boundaries; every chunk is tagged with its clause label so a retrieved answer traces to the label
- **Government/regulatory text** - hierarchical Title -> Part -> Subpart -> § splitting with a tighter token budget (to prioritize precision over breadth); chunks are flagged when they contain a statutory citation
- **Technical manuals** - split on markdown headings; code blocks are kept as atomic units regardless of token budget, since a broken code sample is worse than an oversized chunk
- **Financial reports** - tables are detected and extracted as atomic chunks (never split mid-table); narrative text around them is chunked normally
- **Narrative prose** - paragraph-packing that respects chapter boundaries as hard breaks; optionally supports embedding-breakpoint chunking (split where meaning actually shifts between paragraphs) via a pluggable `embed_fn`, falling back to plain packing when none is provided
- **Default** - recursive paragraph -> sentence splitting, used for anything unclassified or without a dedicated strategy

### Multi-format ingestion

`ragchunk/loaders.py` reads `.txt`, `.md`, `.pdf`, and `.docx` directly, dispatched by file extension. PDF text is extracted page-by-page; DOCX ingestion also pulls table content out of Word tables (not just paragraph text) so it reaches the financial chunker's table detector. PDF/DOCX support are optional dependencies, so if `pypdf` or `python-docx` aren't installed, reading that file type raises a clear, actionable error instead of a stack trace; `.txt`/`.md` files always work with zero dependencies.

### Real token accounting

Chunk sizes are budgeted against actual token counts via `tiktoken`, not a word-count guess, so `max_tokens` settings per chunker translate to real retrieval-relevant chunk sizes. Falls back automatically to a word-count approximation if `tiktoken` isn't installed, so the pipeline never hard-requires it.

### Full provenance metadata

Every chunk carries `doc_type`, `section_path`, `chunk_index`, `token_estimate`, and flexible `extra_metadata` dictionary (which feature: citation flags, code-block/table markers, which coarse section a mixed document's chunk came from). This is what makes a retrieved chunk traceable back to exactly where it came from, rather than an anonymous chunk of text.

## Project structure


 
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
sample_docs/            One example per document type, including a mixed-type sample
tests/                  Classifier, pipeline, and utility test suites
main.py                 Command-line entry point
requirements.txt        All dependencies are optional; see below
```
## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The core pipeline runs with **zero required dependencies**, heuristic classification and every chunking strategy are purely part of the standard library. Everything in `requirements.txt` is optional and the pipeline degrades gracefully without it;

| Package | Enables | Without it |
|---|---|---|
| `tiktoken` | Exact token counts | Falls back to a word-count approximation |
| `pypdf` | Reading `.pdf` files | Reading a PDF raises a clear `ImportError` telling you to install it |
| `python-docx` | Reading `.docx` files| Same graceful error as above, but for Word files |
| `google-genai` (or any other one of your choosing) | LLM fallback classification for ambiguous documents | Ambiguous documents are still classified, just via the best heuristic guess, with no escalation |

### Configuration

The LLM classification fallback is provider-agnostic by design, so `HybridClassifier` and `AdaptiveChunkingPipeline` both accept any `llm_classify_fn: Callable[[str], str]`. `main.py`'s `build_llm_classify_fn()` shows a working example wired to Google AI Studio (Gemini); you can swap the client construction there for Anthropic or OpenAI if using a different provider. Set the provider's API key as an environment variable before running with `--use-llm`:

```bash
# example for the Gemini wiring shipped in main.py
export GOOGLE_API_KEY=AIza...        # $env:GOOGLE_API_KEY = "AIza..." on Windows PowerShell
```

## Running the pipeline

```bash
# a single file (any of .txt / .md / .pdf / .docx)
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
# or, with pytest installed:
pytest tests/
```

## What's implemented

- Hybrid heuristic + pluggable-LLM document classification
- Section-level classification for ducments that mix types, with automic fallback to whole-document classification for uniform documents
- Six chunking strategies: legal, government/regulatory, technical, financial (table-aware), narrative (with optional semantic mode), and a generic fallback
- Multi-format ingestion: `.txt`, `.md`, `.pdf`, `.docx`
- Real token budgeted chunk sizing via `tiktoken`, with graceful fallback
- Full chunk-level provenance metadata (section path, citation/table/code flags, token counts)

## Planned additions

- **Evaluation and calibration** - labeled eval sets per document type, precision/recall tracking for the classifier, and tuning the heuristic rule weights and confidence thresholds against real data instead of hand-picked values
- **Retrieval-quality evaluation** - a per-type Q&A eval harness so chunking changes can be measured by actual retrieval performance, not just chunk-correctness assertions
- **Vector store integration** - pluggable embedding + storage (Chroma / Qdrant / pgvector), following the same swappable-function pattern already used for LLM classification and narrative embeddings
- **Parent-child retrieval indexing** - index small, precise chunks for retrieval but expand to their parent section for generation context, pairing naturally with the `section_path` hierarchy the chunkers already produce
- **Batch processing with checkpointing** for large document sets, plus a caching layer so re-running the pipeline on unchanged documents doesn't re-classify or re-chunk
- **CLI wiring for the semantic narrative chunker** - `embed_fn` is currently only usable programatically; a `--embed` flag would make it available from the command line
- **A feedback loop** - logging retrieval performance per chunk and periodically re-tuning chunk size/overlap per type based on observed quality, rather than static defaults