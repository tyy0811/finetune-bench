"""Training data privacy audit: redaction markers, PII scan, duplicates."""

from __future__ import annotations

import re
from collections import Counter

# CFPB redaction patterns — verified empirically from complaint narratives.
# XX/XX/XXXX must match before XXXX to avoid partial overlap.
_REDACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("XX/XX/XXXX", re.compile(r"XX/XX/XXXX")),
    ("XXXXXXXX", re.compile(r"X{8,}")),
    ("XXXX", re.compile(r"(?<!/)(?<!X)X{4}(?!X)(?!/)")),
]


def detect_redaction_markers(texts: list[str]) -> dict:
    """Scan texts for CFPB-style redaction markers.

    Returns dict with 'count' (total markers found) and
    'patterns_found' (list of distinct pattern names seen).
    """
    total = 0
    seen_patterns: set[str] = set()

    for text in texts:
        for name, pattern in _REDACTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                total += len(matches)
                seen_patterns.add(name)

    return {
        "count": total,
        "patterns_found": sorted(seen_patterns),
    }
