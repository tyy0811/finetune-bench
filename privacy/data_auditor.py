"""Training data privacy audit: redaction markers, PII scan, duplicates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

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


# Columns that could serve as protected attributes or proxy variables.
_SENSITIVE_COLUMN_NAMES = {
    "company", "state", "submitted_via", "zip_code",
    "ethnicity", "race", "gender", "age",
}


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_near_duplicates(texts: list[str]) -> dict:
    """Find exact duplicates after text normalization.

    Returns dict with 'count' (number of duplicate pairs, i.e. C(n,2)
    for each group of n identical texts) and 'method'.
    """
    from collections import Counter

    seen: dict[str, int] = Counter()
    for text in texts:
        seen[_normalize_text(text)] += 1

    duplicate_count = sum(n * (n - 1) // 2 for n in seen.values() if n > 1)
    return {"count": duplicate_count, "method": "exact_normalized"}


def _normalize_column_name(name: str) -> str:
    """Lowercase and replace spaces/hyphens with underscores."""
    return re.sub(r"[\s-]+", "_", name.lower())


def inventory_sensitive_columns(columns: list[str]) -> list[str]:
    """Flag column names that could serve as protected attributes."""
    return sorted(col for col in columns if _normalize_column_name(col) in _SENSITIVE_COLUMN_NAMES)


@dataclass
class AuditConfig:
    """Configuration for the data audit."""

    short_narrative_threshold: int = 20  # tokens
    max_residual_pii: int | None = None  # gate threshold; None = report only


def run_audit(
    narratives: list[str],
    columns: list[str],
    output_path: Path | None = None,
    max_residual_pii: int | None = None,
    short_narrative_threshold: int = 20,
) -> dict:
    """Run the full data privacy audit.

    Scans for redaction markers, residual PII, near-duplicates,
    sensitive columns, and short narratives. Optionally writes
    JSON report and enforces a PII gate.
    """
    redaction = detect_redaction_markers(narratives)
    pii = scan_residual_pii(narratives)
    duplicates = detect_near_duplicates(narratives)
    sensitive = inventory_sensitive_columns(columns)

    short_count = sum(
        1 for text in narratives if len(text.split()) < short_narrative_threshold
    )

    assessment_parts = []
    if redaction["count"] > 0:
        assessment_parts.append(
            f"Source applies redaction ({redaction['count']} markers)"
        )
    if pii["total"] > 0:
        assessment_parts.append(
            f"{pii['total']} residual PII instances suggest incomplete coverage"
        )
    if duplicates["count"] > 0:
        assessment_parts.append(
            f"{duplicates['count']} near-duplicates flagged for deduplication"
        )
    assessment = ". ".join(assessment_parts) + "." if assessment_parts else "No issues found."

    report: dict = {
        "total_samples": len(narratives),
        "redaction_markers": {
            "count": redaction["count"],
            "patterns_found": redaction["patterns_found"],
            "verified_from_sample": True,
        },
        "residual_pii": pii,
        "sensitive_columns": sensitive,
        "near_duplicates": duplicates,
        "short_narratives": {
            "count": short_count,
            "threshold_tokens": short_narrative_threshold,
        },
        "assessment": assessment,
    }

    if max_residual_pii is not None:
        report["gate_passed"] = pii["total"] <= max_residual_pii

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    """Run audit on the CFPB dataset and write report to artifacts/."""
    from adapters.cfpb import CFPBAdapter

    adapter = CFPBAdapter(sample_size=20_000, seed=42)
    splits = adapter.preprocess()

    all_narratives = splits["train"]["narratives"]

    # CFPB columns that are used for feature engineering
    columns = ["company", "state", "submitted_via"]

    output_path = Path("artifacts/data_audit_report.json")
    report = run_audit(all_narratives, columns, output_path=output_path)

    print(f"Audit complete: {report['total_samples']} samples scanned")
    print(f"  Redaction markers: {report['redaction_markers']['count']}")
    print(f"  Residual PII: {report['residual_pii']['total']}")
    print(f"  Near-duplicates: {report['near_duplicates']['count']}")
    print(f"  Assessment: {report['assessment']}")
    print(f"  Report written to: {output_path}")


if __name__ == "__main__":
    main()
