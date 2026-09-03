from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..classifier import HybridClassifier
from ..loaders import load_document
from .dataset import LabeledExample

CONFIDENCE_SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(13)]
MARGIN_SWEEP = [round(0.05 + 0.05 * i, 2) for i in range(8)]

@dataclass
class CalibrationResult: 
    confidence_threshold: float
    margin_threshold: float
    heuristic_only_accuracy: float
    escalation_rate: float
    n_docs: int

def run_calibration_sweep(
        dataset: List[LabeledExample], base_dir: Optional[str] = None
) -> List[CalibrationResult]:
    examples = [e for e in dataset if not e.is_mixed]
    texts = []
    for e in examples:
        doc_path = str(Path(base_dir) / e.path) if base_dir else e.path
        texts.append(load_document(doc_path))

    results: List[CalibrationResult] = []
    for conf_t in CONFIDENCE_SWEEP:
        for margin_t in MARGIN_SWEEP:
            classifier = HybridClassifier(
                llm_classify_fn=None,
                confidence_threshold=conf_t,
                margin_threshold=margin_t,
            )

            would_escalate = 0
            correct_no_escalate = 0
            total_no_escalate = 0

            for example, text in zip(examples, texts):
                classification = classifier.classify(text)
                is_ambiguous = classification.signals.get("fallback_reason") == "ambiguous_no_llm_confirmed"
                if is_ambiguous:
                    would_escalate += 1
                else:
                    total_no_escalate += 1
                    if classification.doc_type.value == example.expected_type:
                        correct_no_escalate += 1

            n = len(examples)
            results.append(CalibrationResult(
                confidence_threshold=conf_t,
                margin_threshold=margin_t,
                heuristic_only_accuracy=(correct_no_escalate / total_no_escalate) if total_no_escalate else 1.0,
                escalation_rate=(would_escalate / n) if n else 0.0,
                n_docs=n,
            ))
    return results

def recommend_thresholds(results: List[CalibrationResult]) -> CalibrationResult:
    perfect = [r for r in results if r.heuristic_only_accuracy == 1.0]
    candidates = perfect if perfect else results
    return min(candidates, key=lambda r: (r.escalation_rate, -r.confidence_threshold))
