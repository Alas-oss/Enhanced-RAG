from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .calibration import CalibrationResult
from .classifier_eval import ClassifierEvalReport
from .retrieval_eval import RetrievalEvalReport


def print_classifier_report(report: ClassifierEvalReport) -> None:
    print("\n Classifier Evaluation ")
    print(f"Accuracy: {report.correct}/{report.total} ({report.accuracy:.1%})")

    print("\nPer-class metrics:")
    print(f"{'type':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for cls, m in sorted(report.per_class_metrics.items()):
        print(f"{cls:<24}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>10.0f}")

    if report.mismatch:
        print("\nMismatch:")
        for m in report.mismatch:
            print(f"  {m['path']}: expected {m['expected']}, got {m['predicted']} (confidence {m['confidence']:.2f})")
    else:
        print("\nNo mismatch on non-mixed documents.")

    if report.mixed_doc_results:
        print("\nMixed-document detection:")
        for r in report.mixed_doc_results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['path']}: expected {r['expected_section_types']}, detected {r['detected_types']}")


def print_retrieval_report(report: RetrievalEvalReport) -> None:
    print("\n Retrieval Evaluation ")
    print(f"Hit rate (top-k): {report.hits}/{report.total} ({report.hit_rate:.1%})")
    print(f"Mean Reciprocal Rank: {report.mean_reciprocal_rank:.2f}")

    print("\nPer-type hit rate:")
    for t, rate in sorted(report.per_type_hit_rate.items()):
        print(f"  {t:<24}{rate:.1%}")

    misses = [d for d in report.details if not d["hit"]]
    if misses:
        print("\nMisses:")
        for d in misses:
            print(
                f"  \"{d['question']}\" -- expected section containing "
                f"'{d['expected_section_contains']}', top result was '{d['top_result_section']}'"
            )
    else:
        print("\nNo misses.")


def print_calibration_recommendation(recommended: CalibrationResult, all_results: List[CalibrationResult]) -> None:
    print("\n Calibration ")
    print(f"Swept {len(all_results)} threshold combinations (heuristic-only, no LLM).")
    print(
        f"Recommended: confidence_threshold={recommended.confidence_threshold}, "
        f"margin_threshold={recommended.margin_threshold}"
    )
    print(
        f"  -> {recommended.heuristic_only_accuracy:.1%} accuracy on non-escalated docs, "
        f"{recommended.escalation_rate:.1%} of the labeled set would escalate to LLM at this setting"
    )
    print("  Current defaults in classifier.py: confidence_threshold=0.55, margin_threshold=0.15")


def export_full_report(
    classifier_report: ClassifierEvalReport,
    retrieval_report: RetrievalEvalReport,
    recommended_calibration: CalibrationResult,
    output_path: str,
) -> None:
    payload: Dict[str, Any] = {
        "classifier_eval": {
            "accuracy": classifier_report.accuracy,
            "total": classifier_report.total,
            "correct": classifier_report.correct,
            "per_class_metrics": classifier_report.per_class_metrics,
            "confusion": classifier_report.confusion,
            "mismatch": classifier_report.mismatch,
            "mixed_doc_results": classifier_report.mixed_doc_results,
        },
        "retrieval_eval": {
            "hit_rate": retrieval_report.hit_rate,
            "mean_reciprocal_rank": retrieval_report.mean_reciprocal_rank,
            "per_type_hit_rate": retrieval_report.per_type_hit_rate,
            "details": retrieval_report.details,
        },
        "calibration": {
            "recommended_confidence_threshold": recommended_calibration.confidence_threshold,
            "recommended_margin_threshold": recommended_calibration.margin_threshold,
            "heuristic_only_accuracy_at_recommendation": recommended_calibration.heuristic_only_accuracy,
            "escalation_rate_at_recommendation": recommended_calibration.escalation_rate,
        },
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
