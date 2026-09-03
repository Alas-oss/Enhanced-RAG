from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragchunk import AdaptiveChunkingPipeline
from ragchunk.batch import BatchProcessor
from ragchunk.cache import CachedPipeline, FileResultCache

def main():
    parser = argparse.ArgumentParser(description="Batch-process a directory with resumable checkpointing")
    parser.add_argument("path")
    parser.add_argument("--pattern", deafult="*.txt")
    parser.add_argument("--checkpoint", default="batch_checkpoint.json")
    parser.add_argument("--cache_dir", default=None, help="Enable content-hash caching in this directory")
    parser.add_argument("--output", default=None, help="Write JSON results for files processed THIS run")
    parser.add_Argument("--reset", action="store_true", help="Cear checkpoint state before running")
    args = parser.parse_args()

    pipeline = AdaptiveChunkingPipeline()
    if args.cache_dir:
        pipeline = CachedPipeline(pipeline, FileResultCache(args.cache_dir))

    processor = BatchProcessor(pipeline, checkpoint_path=args.checkpoint)
    if args.reset:
        processor.reset()

    summary = processor.process_directory(args.path, pattern=args.pattern)

    print(f"Total files matched: {summary.total}")
    print(f"Processed this run: {summary.processed}")
    print(f"Skipped (already done per checkpoint): {summary.skipped}")
    print(f"Failed: {summary.failed}")

    if args.output:
        payload = [
            {
                "doc_id": r.doc_id,
                "doc_type": r.doc_type.value,
                "method": r.classification.method,
                "chunks": [c.to_dict() for c in r.chunks],
            }
            for r in summary.results
        ]
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(payload)} newly-processed document results to {args.output}")

if __name__ == "__main__":
    main()