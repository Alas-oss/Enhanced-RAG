from __future__ import annotations

import argparse
from pathlib import Path

from ragchunk.eval.calibration import recommend_thresholds, run_calibration_sweep
from ragchunk.eval.classifier_eval import run_classifier_eval
from ragchunk.eval.dataset import load_labeled_dataset, load_qa_dataset
from ragchunk.eval.report import (
    export_full_report,
    print_calibration_recommendation,
    print_classifier_report,
    print_retrieval_report,
)
from ragchunk.eval.retrieval_eval import run_retrieval_eval

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Evaluate the adaptive chunking pipeline")
    parser.add_argument("--labeled-dataset", default=str(PROJECT_ROOT / "eval_data" / "labeled_dataset.json"))
    parser.add_argument("--qa-dataset", default=str(PROJECT_ROOT / "eval_data" / "qa_dataset.json"))
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--output", default=None, help="Write a full JSON report to this path")
    args = parser.parse_args()

    labeled_dataset = load_labeled_dataset(args.labeled_dataset)
    qa_dataset = load_qa_dataset(args.qa_dataset)

    classifier_report = None
    retrieval_report = None
    recommended = None
    all_calibration_results = []


    if not args.skip_classifier:
        classifier_report = run_classifier_eval(labeled_dataset, base_dir=str(PROJECT_ROOT))
        print_classifier_report(classifier_report)

    if not args.skip_retrieval:
        retrieval_report = run_retrieval_eval(qa_dataset, base_dir=str(PROJECT_ROOT))
        print_retrieval_report(retrieval_report)


    if not args.skip_calibration:
        all_calibration_results = run_calibration_sweep(labeled_dataset, base_dir=str(PROJECT_ROOT))
        recommended = recommend_thresholds(all_calibration_results)
        print_calibration_recommendation(recommended, all_calibration_results)

    if args.output and classifier_report and retrieval_report and recommended:
        export_full_report(classifier_report, retrieval_report, recommended, args.output)
        print(f"\nWrote full report to {args.output}")

if __name__ == "__main__":
    main()