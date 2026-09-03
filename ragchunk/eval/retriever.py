from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]

class TfidfRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.n_docs = len(documents)

        doc_tokens = [tokenize(doc) for doc in documents]
        self.df: Counter = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

        self.doc_vectors = [self._vectorize(tokens) for tokens in doc_tokens]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        return {term: count * self._idf(term) for term, count in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def rank(self, query: str) -> List[Tuple[int, float]]:
        query_vec = self._vectorize(tokenize(query))
        scores = [(i, self._cosine(query_vec, doc_vec)) for i, doc_vec in enumerate(self.doc_vectors)]
        return sorted(scores, key=lambda x: x[1], reverse=True)