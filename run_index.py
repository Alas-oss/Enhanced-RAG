from __future__ import annotations

import argparse 
import sys

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.embeddings import build_gemini_embed_fn, build_hashed_fallback_embed_fn
from ragchunk.store import ChromaVectorStore, InMemoryVectorStore, Indexer

def main():
    parser = argparse.ArgumentParser(description="Index documents into a vector store and query them")
    parser.add_argument("path", hrelp="File or directory to index")
    parser.add_argument("--dir", action="store_true", help="Treat `path` as a directory")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--store", choices=["memory", "chroma"], default="memory")
    parser.add_argument("--persist-dir", default=None, help="Chroma persistence directory (--sotre chroma only)")
    parser.add_argument("--query", default=None, help="Run a query after indexing")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-parent-expansion", action="store_true")
    args = parser.parse_args()

    embed_fn = build_gemini_embed_fn()
    if embed_fn is None:
        print(
            "[info] No real embedding provider configured -- using a local hashed " \
            "bag-of-words stand-in for demonstration. This is NOt semantic search. " \
            "Set GOOGLE_API_KEY and install google-genai for real embeddings.",
            file=sys.stderr,
        )
        embed_fn = build_hashed_fallback_embed_fn()

    if args.store == "chroma":
        vector_store = ChromaVectorStore(persist_directory=args.persis_dir)
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
    print(f"Indexed {count} chunks.")

    if args.query:
        results = indexer.query(args.query, top_k=args.top_k, expand_to_parent=not args.no_parent_expansion)
        print(f'\nTop {len(results)} results for: "{args.query}"\n')
        for i, r in enumerate(results, start=1):
            chunk = r["chunk"]
            print(f"{i}. [{chunk.doc_type}] {chunk.section_path} (score {r['score']:.3f})")
            print(f"    {chunk.text[:100].strip().replace(chr(10), ' ')}...")
            if "parent_text" in r and r["parent_text"] != chunk.text:
                print(f"    (parent context available: {len(r['parent_text'])} chars)")
            print()

if __name__ == "__main__":
    main()