from __future__ import annotations

import argparse
import json
import sys

from ragchunk.pipeline import AdaptiveChunkingPipeline

def build_llm_classify_fn():
    try:
        from google import genai

        client = genai.Client()

        def classify_fn(prompt: str) -> str:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return resp.text.strip()

        return classify_fn
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Adaptive RAG chunking pipeline")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument("--dir", action="store_true", help="Treat `path` as a directory")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern when --dir is set")
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM fallback classification for ambiguous documents "
        "(required `pip install gemini` and GEMINI_API_KEY set)",
    )
    args = parser.parse_args()
    llm_fn = build_llm_classify_fn() if args.use_llm else None
    if args.use_llm and llm_fn is None:
        print(
            "[warn] --use-llm was set but the gemini client could not be " \
            "initialized (missing package or API key). Continuing with " \
            "heuristic-only classification.",
            file=sys.stderr,
        )

    pipeline = AdaptiveChunkingPipeline(llm_classify_fn=llm_fn)

    if args.dir:
        results = pipeline.process_directory(args.path, pattern=args.pattern)
    else:
        results = [pipeline.process_file(args.path)]

    output_payload = []
    for result in results:
        print(f"\n=== {result.doc_id} ===")
        print(f"  doc_type   : {result.doc_type.value}")
        print(f"  method     : {result.classification.method}")
        print(f"  confidence : {result.classification.confidence:.2f}")
        print(f"  chunks     : {len(result.chunks)}")
        for c in result.chunks[:3]:
            preview = c.text[:80].replace("\n", " ")
            print(f"    - [{c.section_path}] ({c.token_estimate} tok) {preview}...")
        if len(result.chunks) > 3:
            print(f"    ... and {len(result.chunks) - 3} more")

        output_payload.append({
            "doc_id": result.doc_id,
            "doc_type": result.doc_type.value,
            "classification": {
                "confidence": result.classification.confidence,
                "method": result.classification.method,
            },
            "chunks": [c.to_dict() for c in result.chunks],
        })

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nWrote results to {args.output}")

if __name__=="__main__":
    main()