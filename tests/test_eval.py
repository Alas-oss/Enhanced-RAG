import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchunk.eval.calibration import recommend_thresholds, run_calibration_sweep
from ragchunk.eval.classifier_eval import run_classifier_eval
from ragchunk.eval.dataset import load_labeled_dataset, load_qa_dataset
from ragchunk.eval.retrieval_eval import run_retrieval_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELED_DATASET_PATH = PROJECT_ROOT / "eval_data" / "labeled_dataset.json"
QA_DATASET_PATH = PROJECT_ROOT / "eval_data" / "qa_dataset.json"

def test_classifier_eval_perfect_accuracy_on_labeled_set():
    dataset = load_labeled_dataset(str(LABELED_DATASET_PATH))
    report = run_classifier_eval(dataset, base_dir=str(PROJECT_ROOT))

    assert report.accuracy == 1.0
    assert len(report.mismatch) == 0
    for cls, metrics in report.per_class_metrics.items():
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

def test_classifier_eval_detects_mixed_document_correctly():
    dataset = load_labeled_dataset(str(LABELED_DATASET_PATH))
    report = run_classifier_eval(dataset, base_dir=str(PROJECT_ROOT))

    assert len(report.mixed_doc_results) == 1
    mixed_result = report.mixed_doc_results[0]
    assert mixed_result["passed"] is True
    assert "technical_manual" in mixed_result["detected_types"]

def test_retrieval_eval_hits_expected_chunks():
    qa_dataset = load_qa_dataset(str(QA_DATASET_PATH))
    report = run_retrieval_eval(qa_dataset, base_dir=str(PROJECT_ROOT))

    assert report.total == len(qa_dataset)
    assert report.hit_rate >= 0.9

def test_calibration_sweep_returns_results_and_recommendation():
    dataset = load_labeled_dataset(str(LABELED_DATASET_PATH))
    results = run_calibration_sweep(dataset, base_dir=str(PROJECT_ROOT))

    assert len(results) > 0
    recommended = recommend_thresholds(results)
    assert 0.0 <= recommended.confidence_threshold <= 1.0
    assert 0.0 <= recommended.margin_threshold <= 1.0
    assert recommended.heuristic_only_accuracy == 1.0

if __name__ == "__main__":
    test_classifier_eval_perfect_accuracy_on_labeled_set()
    test_classifier_eval_detects_mixed_document_correctly()
    test_retrieval_eval_hits_expected_chunks()
    test_calibration_sweep_returns_results_and_recommendation()
    print("All eval harness tests passed.")