from __future__ import annotations

import hashlib
from typing import Callable, List, Optional

def build_gemini_embed_fn() -> Optional[Callable[[List[str]], List[List[float]]]]:
    try: 
        from google import genai

        client = genai.Client()

        def embed_fn(texts: List[str]) -> List[List[float]]:
            result = client.models.embed_content(model="text-embedding-004", contents=texts)
            return [e.values for e in result.embeddings]

        return embed_fn
    except Exception:
        return None

def build_hashed_fallback_embed_fn(dim: int = 256) -> Callable[[List[str]], List[List[float]]]:
    def embed_fn(texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * dim
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
                vec[idx] += 1.0
            vectors.append(vec)
        return vectors

    return embed_fn