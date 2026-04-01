"""Tests for privacy.data_auditor module."""

import pytest

from privacy.data_auditor import detect_redaction_markers


class TestRedactionMarkerDetection:
    def test_detects_xxxx_pattern(self):
        texts = [
            "I called XXXX bank about my account",
            "On XX/XX/XXXX I made a payment",
            "No redaction here",
        ]
        result = detect_redaction_markers(texts)
        assert result["count"] == 2  # XXXX in text 1 + XX/XX/XXXX in text 2 (no double-count)
        assert "XXXX" in result["patterns_found"]

    def test_detects_date_redaction_pattern(self):
        texts = ["On XX/XX/XXXX they charged me"]
        result = detect_redaction_markers(texts)
        assert "XX/XX/XXXX" in result["patterns_found"]

    def test_no_redaction_markers(self):
        texts = ["This is a normal complaint with no redactions"]
        result = detect_redaction_markers(texts)
        assert result["count"] == 0
        assert result["patterns_found"] == []

    def test_multiple_markers_in_one_text(self):
        texts = ["XXXX sent me to XXXX and charged XXXXXXXX"]
        result = detect_redaction_markers(texts)
        assert result["count"] >= 3

    def test_empty_input(self):
        result = detect_redaction_markers([])
        assert result["count"] == 0
