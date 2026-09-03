from __future__ import annotations

from typing import Callable, Optional

def build_gemini_classify_fn() -> Optional[Callable[[str], str]]:
    try:
        from google import genai

        client = genai.Client()

        def classify_fn(prompt: str) -> str:
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return resp.text.strip()

        return classify_fn
    except Exception:
        return None

def build_gemini_generate_fn(model: str = "gemini-2.5-flash") -> Optional[Callable[[str], str]]:
    try: 
        from google import genai

        client = genai.Client()

        def generate_fn(prompt: str) -> str:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()

        return generate_fn
    except Exception:
        return None