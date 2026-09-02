import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchunk.loaders import load_document
from ragchunk.utils import estimate_tokens

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"


def test_estimate_tokens_returns_positive_int():
    assert estimate_tokens("hello world") > 0
    assert isinstance(estimate_tokens("hello world"), int)


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("A short sentence.")
    long = estimate_tokens("A much longer sentence with quite a few more words in it than the short one.")
    assert long > short


def test_loader_reads_txt_same_as_plain_read():
    path = SAMPLE_DIR / "legal_contract.txt"
    loaded = load_document(str(path))
    direct = path.read_text(encoding="utf-8", errors="ignore")
    assert loaded == direct


def test_loader_pdf_raises_clear_error_without_pypdf():
    try:
        import pypdf  # noqa: F401
        return  # pypdf is installed -- nothing to assert here
    except ImportError:
        pass

    try:
        load_document("nonexistent.pdf")
        assert False, "expected ImportError"
    except ImportError as e:
        assert "pypdf" in str(e)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    test_estimate_tokens_returns_positive_int()
    test_estimate_tokens_scales_with_length()
    test_loader_reads_txt_same_as_plain_read()
    test_loader_pdf_raises_clear_error_without_pypdf()
    print("All Sprint 2 utility tests passed.")
