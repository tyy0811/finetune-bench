"""Tests for privacy.data_auditor module."""

from privacy.data_auditor import detect_redaction_markers, scan_residual_pii


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


class TestResidualPiiScan:
    def test_detects_email(self):
        texts = ["Please contact me at john.doe@example.com for details"]
        result = scan_residual_pii(texts)
        assert result["emails"] >= 1

    def test_detects_phone_number(self):
        texts = ["Call me at 555-123-4567 or (555) 123-4567"]
        result = scan_residual_pii(texts)
        assert result["phones"] >= 1

    def test_detects_ssn_pattern(self):
        texts = ["My SSN is 123-45-6789"]
        result = scan_residual_pii(texts)
        assert result["ssns"] >= 1

    def test_no_pii_in_clean_text(self):
        texts = ["I have a complaint about my mortgage payment being late"]
        result = scan_residual_pii(texts)
        assert result["total"] == 0

    def test_total_is_sum(self):
        texts = [
            "Email john@test.com and call 555-123-4567",
        ]
        result = scan_residual_pii(texts)
        assert result["total"] == result["emails"] + result["phones"] + result["ssns"]

    def test_empty_input(self):
        result = scan_residual_pii([])
        assert result["total"] == 0
