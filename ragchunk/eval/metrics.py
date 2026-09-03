from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

def compute_confusion(pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    for expected, predicted in pairs:
        matrix[(expected, predicted)] += 1
    return dict(matrix)

def precision_recall_f1(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    classes = sorted(set([e for e, _ in pairs] + [p for _, p in pairs]))
    metrics: Dict[str, Dict[str, float]] = {}

    for cls in classes:
        tp = sum(1 for e, p in pairs if e == cls and p == cls)
        fp = sum( 1 for e, p in pairs if e != cls and p == cls)
        fn = sum(1 for e, p in pairs if e == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for e, _ in pairs if e == cls)

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return metrics