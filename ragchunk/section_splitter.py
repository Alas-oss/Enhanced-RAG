from __future__ import annotations

import re
from typing import List, Optional, Tuple

SECTION_BOUNDARY = re.compile(
    r"""(?mx)
    ^\s*(
        \#{1,2}\s+.+ |                 
        ARTICLE\s+[IVXLC\d]+.* |       
        TITLE\s+[\dA-Z]+.* |
        PART\s+\d+.* |    
        Chapter\s+\d+.* | 
        APPENDIX\s+[A-Z\d]+.*  
    )
    """,
    re.IGNORECASE,
)

MIN_SECTION_CHARS = 200

def split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    matches = list(SECTION_BOUNDARY.finditer(text))
    if len(matches) <= 1:
        return [(None, text)]

    raw_sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = match.group(1).strip()[:80]
        raw_sections.append((label, text[start:end]))

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()]
        if preamble.strip():
            raw_sections.insert(0, (None, preamble))

    return _merge_short_sections(raw_sections)

def _merge_short_sections(sections: List[Tuple[Optional[str], str]]) -> List[Tuple[Optional[str], str]]:
    section = list(sections)
    merged: List[Tuple[Optional[str], str]] = []
    i = 0
    while i < len(sections):
        label, sec_text = sections[i]
        if len(sec_text.strip()) < MIN_SECTION_CHARS:
            if merged:
                prev_label, next_text = merged[-1]
                merged[-1] = (prev_label, sec_text + next_text)
            else:
                merged.append((label, sec_text))
        else:
            merged.append((label, sec_text))
        i += 1
    return merged