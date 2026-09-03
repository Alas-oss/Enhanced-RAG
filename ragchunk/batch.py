from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .types import PipelineResult

@dataclass
class BatchRunSummary:
    total: int
    processed: int
    skipped: int
    failed: int
    results: List[PipelineResult] = field(default_factory=list)

class BatchProcessor:
    def __init__(self, pipeline: Any, checkpoint_path: str):
        self.pipeline = pipeline
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint: Dict[str, Dict[str, Any]] = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict[str, Dict[str, Any]]:
        if self.checkpoint_path.exists():
            try:
                return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.write_text(json.dumps(self.checkpoint, indent=2), encoding="utf-8")

    def process_directory(
            self, 
            dir_path: str,
            pattern: str = "*.txt",
            on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> BatchRunSummary:
        files = sorted(Path(dir_path).glob(pattern))
        processed = 0
        skipped = 0
        failed = 0
        results: List[PipelineResult] = []

        for file_path in files:
            key = str(file_path)
            if self.checkpoint_path.get(key, {}).get("status") == "done":
                skipped += 1
                continue

            try:
                result = self.pipeline.process_file(str(file_path))
                results.append(result)
                self.checkpoint[key] = {
                    "status": "done",
                    "doc_type": result.doc_type.value,
                    "chunk_count": len(result.chunks),
                }
                processed += 1
                if on_result:
                    on_result(result)
            except Exception as exc: # noqa: BLE001
                self.checkpoint[key] = {"status": "failed", "error": str(exc)}
                failed += 1
            finally:
                self._save_checkpoint()

        return BatchRunSummary(total=len(files), processed=processed, skipped=skipped, failed=failed, results=results)

    def reset(self) -> None:
        self.checkpoint = {}
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()