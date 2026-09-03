from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..pipeline import AdaptiveChunkingPipeline
from .dataset import LabeledExample
from .metrics import compute_confusion, precision_recall_f1

@dataclass
class ClassifierEvalReport:
    total: int
    correct: int
    accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion: Dict[str, int]
    mismatch: List[Dict[str, Any]]
    mixed_doc_results: List[Dict[str, Any]]

def run_classifier_eval(
        dataset: List[LabeledExample],
        pipeline: Optional[AdaptiveChunkingPipeline] = None,
        base_dir: Optional[str] = None,
) -> ClassifierEvalReport:
    pipeline = pipeline or AdaptiveChunkingPipeline()

    pairs = []
    mismatches: List[Dict[str, Any]] = []
    mixed_results: List[Dict[str, Any]] = []

    for example in dataset:
        doc_path = str(Path(base_dir) / example.path) if base_dir else example.path
        result = pipeline.process_file(doc_path)

        if example.is_mixed:
            detected_types = sorted({c.doc_type for c in result.chunks})
            expected = sorted(example.expected_section_types or [])
            passed = (
                result.classification.method == "section_level_mixed"
                and set (expected).issubset(set(detected_types))
            )
            mixed_results.append({
                "path": example.path,
                "expected_section_types": expected,
                "detected_types": detected_types,
                "method": result.classification.method,
                "passed": passed,
            })
            continue

        predicted = result.doc_type.value
        pairs.append((example.expected_type, predicted))
        if predicted != example.expected_type:
            mismatches.append({
                "path": example.path,
                "expected": example.expected_type,
                "predicted": predicted,
                "confidence": result.classification.confidence,
                "method": result.classification.method,
            })

    correct = sum(1 for e, p in pairs if e == p)
    total = len(pairs)
    accuracy = correct / total if total else 0.0

    confusion_raw = compute_confusion(pairs)
    confusion = {f"{e} -> {p}": count for (e, p), count in confusion_raw.items()}
    per_class = precision_recall_f1(pairs)

    return ClassifierEvalReport(
        total=total,
        correct=correct,
        accuracy=accuracy,
        per_class_metrics=per_class,
        confusion=confusion,
        mismatch=mismatches, 
        mixed_doc_results=mixed_results,
    )