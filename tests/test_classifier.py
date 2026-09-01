import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchunk.classifier import heuristic_classify
from ragchunk.types import DocType
from ragchunk.utils import load_text_file

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"


def test_legal_contract_classified_correctly():
    text = load_text_file(SAMPLE_DIR / "legal_contract.txt")
    result = heuristic_classify(text)
    assert result.doc_type == DocType.LEGAL_CONTRACT
    assert result.confidence > 0.3


def test_government_regulation_classified_correctly():
    text = load_text_file(SAMPLE_DIR / "government_regulation.txt")
    result = heuristic_classify(text)
    assert result.doc_type == DocType.GOVERNMENT_REGULATION
    assert result.confidence > 0.3


def test_technical_manual_classified_correctly():
    text = load_text_file(SAMPLE_DIR / "technical_manual.txt")
    result = heuristic_classify(text)
    assert result.doc_type == DocType.TECHNICAL_MANUAL


def test_narrative_classified_correctly():
    text = load_text_file(SAMPLE_DIR / "narrative.txt")
    result = heuristic_classify(text)
    assert result.doc_type == DocType.NARRATIVE_PROSE


def test_empty_text_falls_back_to_default():
    result = heuristic_classify("   ")
    assert result.doc_type == DocType.DEFAULT
    assert result.confidence == 0.0


if __name__ == "__main__":
    test_legal_contract_classified_correctly()
    test_government_regulation_classified_correctly()
    test_technical_manual_classified_correctly()
    test_narrative_classified_correctly()
    test_empty_text_falls_back_to_default()
    print("All classifier tests passed.")
