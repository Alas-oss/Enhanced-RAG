from __future__ import annotations

import argparse
import json
import sys

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.embeddings import build_gemini_embed_fn
from ragchunk.generation import generate_answer
from ragchunk.llm import build_gemini_generate_fn
from ragchunk.store import ChromaVectorStore, InMemoryVectorStore, Indexer

def main():
    parser = argparse.ArgumentParser(
        description="Index documents and ask questions against them (retrieval + generation)"
    )
    parser.add_argument("path", help="File or directory to index")
    parser.add_argument("--dir", action="store_true", help="Treat `path` as a directory")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--store", choices=["memory", "chroma"], default="memory")
    parser.add_argument("persist-dir", default=None, help="Chroma persistence directory (--store chroma onle)")
    parser.add_argument("--ask", required=True, help="The question to answer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-parent-expansion", action="store_true")
    parser.add_argument("--output", default=None, help="Write the full result (answer + citations) as JSON")
    args = parser.parse_args()

    embed_fn = build_gemini_embed_fn()
    generate_fn = build_gemini_generate_fn()

    if embed_fn is None or generate_fn is None:
        print(
            "[error] run query.py requires a real LLM provider for both embeddings and " \
            "generation -- retrieval alone can fall back to a local stand-in " \
            "(see run_index.py), but generating a real answer cannot. " \
            "Set GOOGLE_API_KEY and install google-genai, or wire in your own " \
            "provider via ragchunk.embeddings / ragchunk.llm.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.store == "chroma":
        vector_store = ChromaVectorStore(persist_directory=args.persist_dir)
    else:
        vector_store = InMemoryVectorStore()

    pipeline = AdaptiveChunkingPipeline()
    indexer = Indexer(
        pipeline, 
        embed_fn=embed_fn,
        vector_store=vector_store,
        build_parent_child=not args.no_parent_expansion,
    )

    if args.dir:
        count = indexer.index_directory(args.path, pattern=args.pattern)
    else:
        count = indexer.index_file(args.path)
        print(f"Indexed {count} chunks.\n")

        retrieved = indexer.query(args.ask, top_k=args.top_k, expand_to_parent=not args.no_parent_expansion)
        result = generate_answer(args.ask, retrieved, generate_fn)

        print(f"Q: {result.question}\n")
        print(f"A: {result.answer}\n")

        if result.citations:
            print("Cited sources:")
            for c in result.citations:
                print(f"    [{c.index}] {c.doc_id} - {c.section_path or 'n/a'} (retrieval score {c.score:3.f})")
        else:
            print("[warn] The answer contains no verifiable citation markers -- treat it with extra caution.")

        cited_ids = {c.chunk_id for c in result.citations}
        unused = [s for s in result.sources_provided if s.chunk_id not in cited_ids]
        if unused:
            print(f"\n({len(unused)} retrieved source(s) were provided to the model but no cited in the answer)")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nWrote full result to {args.output}")

if __name__ == "__main__":
    main()