from __future__ import annotations

import re
from pathlib import Path

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

def estimate_tokens(text: str) -> int:
    if _ENCODING is not None:
        try:
            return max(1, len(_ENCODING.encode(text)))
        except Exception:
            pass
        
    words = len(text.split())
    return max(1, int(words * 1.3))

def load_text_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore")

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()