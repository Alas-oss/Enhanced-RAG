from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from .types import ClassificationResult, DocType

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15

HEURISTIC_RULES: Dict[DocType, List[Tuple[str, float]]] = {
    DocType.LEGAL_CONTRACT: [
        (r"\bWHEREAS\b", 3.0),
        (r"\bhereinafter\b", 2.5),
        (r"\bthe (Parties|Party|Contractor|Licensee|Licensor)\b", 2.0),
        (r"\bindemnif(y|ication)\b", 2.5),
        (r"\bgoverning law\b", 2.0),
        (r"\bin witness whereof\b", 3.0),
        (r"\bshall\s+(not\s+)?be\s+(liable|entitled|responsible)\b", 2.0),
        (r"^\s*(ARTICLE|Article)\s+[IVXLC\d]+", 2.5),
        (r"\bterms and conditions\b", 1.0),
        (r"\bthis agreement\b", 2.0),
    ],
    DocType.GOVERNMENT_REGULATION: [
        (r"\b\d{1,3}\s*U\.?S\.?C\.?\s*§?\s*\d+", 3.5),
        (r"\bC\.?F\.?R\.?\b", 3.0),
        (r"\bpursuant to\b", 2.0),
        (r"\bFederal Register\b", 3.0),
        (r"\bshall promulgate\b", 2.5),
        (r"^\s*(TITLE|PART|Subpart|§)\s*[\dA-Z]", 2.5),
        (r"\bthe Secretary (shall|may)\b", 2.0),
        (r"\beffective date\b", 1.0),
        (r"\bNotice of Proposed Rulemaking\b", 3.0),
    ],
    DocType.TECHNICAL_MANUAL: [
        (r"```", 3.0),
        (r"\bAPI\b", 1.5),
        (r"\bfunction\s+\w+\s*\(", 2.0),
        (r"\bparameters?:\s*$", 1.5),
        (r"^\s*#{1,4}\s+\S", 2.0),  # markdown headings
        (r"\breturns?:\s*$", 1.0),
        (r"\binstall(ation)?\b", 1.0),
        (r"\bconfig(uration)?\b", 1.0),
        (r"\bstep \d+\b", 1.5),
    ],
    DocType.FINANCIAL_REPORT: [
        (r"\b(revenue|EBITDA|net income|balance sheet)\b", 2.5),
        (r"\bfiscal (year|quarter)\b", 2.0),
        (r"\$\s?[\d,]+(\.\d+)?\s?(million|billion|M|B)\b", 2.5),
        (r"\b10-K\b|\b10-Q\b", 3.0),
        (r"\bshareholders?\b", 1.5),
        (r"\bcash flow\b", 2.0),
    ],
    DocType.NARRATIVE_PROSE: [
        (r'"[A-Z][^"]{10,}"', 1.0),  # quoted dialogue
        (r"\bchapter \d+\b", 2.5),
        (r"\bonce upon a time\b", 3.0),
        (r"\bhe said\b|\bshe said\b", 1.5),
    ],
}

def _score_text(text: str) -> Dict[DocType, float]:
    scores: Dict[DocType, float] = {dt: 0.0 for dt in HEURISTIC_RULES}
    for doc_type, rules in HEURISTIC_RULES.items():
        for pattern, weight in rules:
            matches = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
            if matches:
                count = min(len(matches), 5)
                scores[doc_type] += weight * (1 + 0.3 * (count - 1))
    return scores

def _normalize_scores(scores: Dict[DocType, float]) -> Dict[DocType, float]:
    total = sum(scores.values())
    if total <= 0:
        return {dt: 0.0 for dt in scores}
    return {dt: s / total for dt, s in scores.items()}

def heuristic_classify(text: str) -> ClassificationResult:
    sample = text[:8000]
    raw_scores = _score_text(sample)
    norm_scores = _normalize_scores(raw_scores)

    ranked = sorted(norm_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score == 0.0:
        return ClassificationResult(
            doc_type=DocType.DEFAULT,
            confidence=0.0,
            method="heuristic",
            signals={"raw_score": raw_scores, "normalized": norm_scores},
        )

    margin = top_score - second_score
    confidence = min(1.0, top_score * 0.7 + margin * 1.5)

    return ClassificationResult(
        doc_type=top_type,
        confidence=confidence,
        method="heuristic",
        signals={
            "raw_score": raw_scores,
            "normalized": norm_scores,
            "margin": margin,
        },
    )

LLM_CLASSIFIER_PROMPT = """Classify the following document excerpt into exactly one category:
- legal_contract
- government_regulation
- technical_manual
- financial_report
- narrative_prose
- default (use this if none of the above clearly fit)

Respond with ONLY the category label, nothing else.

Document excerpt:
---
{excerpt}"""

class LLMClassifierClient:
    @staticmethod
    def not_configured(prompt: str) -> str:
        raise RuntimeError(
            "No LLM classify_fn configured. Pass `llm_classify_fn` to " \
            "HybridClassifier, or rely on heuristic-only classification."
        )

class HybridClassifier:
    def __init__(self, 
            llm_classify_fn: Optional[Callable[[str], str]] = None,
            confidence_threshold: float = CONFIDENCE_THRESHOLD,
            margin_threshold: float = MARGIN_THRESHOLD):
        self.llm_classify_fn = llm_classify_fn
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def classify(self, text: str) -> ClassificationResult:
        result = heuristic_classify(text)
        margin = result.signals.get("margin", 0.0)

        is_ambiguous = (
            result.confidence < self.confidence_threshold
            or margin < self.margin_threshold
        )

        if not is_ambiguous:
            return result

        if self.llm_classify_fn is None:
            result.signals["fallback_reason"] = "ambiguous_no_llm_configured"
            return result

        excerpt = text[:3000]
        prompt = LLM_CLASSIFIER_PROMPT.format(excerpt=excerpt)
        try:
            raw_label = self.llm_classify_fn(prompt).strip().lower()
            doc_type = DocType(raw_label) if raw_label in DocType._value2member_map_ else DocType.DEFAULT
            return ClassificationResult(
                doc_type=doc_type,
                confidence=0.9,
                method="llm",
                signals={"heuristic_result": result.signals, "raw_label": raw_label},
            )
        except Exception as exc: # noqa: BLE001
            result.signals["fallback_reason"] = f"llm_call_failed: {exc}"
            result.method = "heuristic_after_llm_failure"
            return result