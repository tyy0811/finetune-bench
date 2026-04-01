"""Training data privacy audit: redaction markers, PII scan, duplicates."""

from __future__ import annotations

import re

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


# Regex-based PII patterns for residual detection.
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def scan_residual_pii(texts: list[str]) -> dict:
    """Regex-based scan for PII that survived source-level redaction.

    Returns dict with counts per PII type and a 'total' field.
    """
    emails = 0
    phones = 0
    ssns = 0

    for text in texts:
        emails += len(_EMAIL_RE.findall(text))
        phones += len(_PHONE_RE.findall(text))
        ssns += len(_SSN_RE.findall(text))

    return {
        "emails": emails,
        "phones": phones,
        "ssns": ssns,
        "total": emails + phones + ssns,
    }
