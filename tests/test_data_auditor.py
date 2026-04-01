"""Tests for privacy.data_auditor module."""

import json
import tempfile
from pathlib import Path

from privacy.data_auditor import (
    detect_near_duplicates,
    detect_redaction_markers,
    inventory_sensitive_columns,
    run_audit,
    scan_residual_pii,
)


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


class TestNearDuplicateDetection:
    def test_exact_duplicates_detected(self):
        texts = ["complaint about fees", "complaint about fees", "different complaint"]
        result = detect_near_duplicates(texts)
        assert result["count"] >= 1

    def test_normalized_duplicates_detected(self):
        texts = [
            "Complaint about  fees!",
            "complaint about fees",
        ]
        result = detect_near_duplicates(texts)
        assert result["count"] >= 1

    def test_triple_duplicate_counts_pairs(self):
        texts = ["same text", "same text", "same text"]
        result = detect_near_duplicates(texts)
        assert result["count"] == 3  # C(3,2) = 3 pairs

    def test_no_duplicates(self):
        texts = ["first complaint", "second complaint", "third complaint"]
        result = detect_near_duplicates(texts)
        assert result["count"] == 0

    def test_method_is_exact_normalized(self):
        texts = ["a", "b"]
        result = detect_near_duplicates(texts)
        assert result["method"] == "exact_normalized"


class TestSensitiveColumnInventory:
    def test_flags_known_sensitive_columns(self):
        columns = ["narrative", "company", "state", "submitted_via", "product"]
        result = inventory_sensitive_columns(columns)
        assert "company" in result
        assert "state" in result
        assert "submitted_via" in result

    def test_does_not_flag_non_sensitive(self):
        columns = ["narrative", "product"]
        result = inventory_sensitive_columns(columns)
        assert "narrative" not in result
        assert "product" not in result

    def test_empty_columns(self):
        result = inventory_sensitive_columns([])
        assert result == []


class TestFullAudit:
    def _sample_data(self):
        """Synthetic CFPB-like data with known PII and redaction markers."""
        return {
            "narratives": [
                "XXXX bank charged me on XX/XX/XXXX without notice",
                "I called XXXX about my mortgage and they said contact john@test.com",
                "My credit card from XXXX has wrong charges",
                "I called XXXX about my mortgage and they said contact john@test.com",
                "Normal complaint with no special content",
            ],
            "columns": ["narrative", "company", "state", "submitted_via", "product"],
        }

    def test_audit_produces_complete_report(self):
        data = self._sample_data()
        report = run_audit(data["narratives"], data["columns"])
        assert "redaction_markers" in report
        assert "residual_pii" in report
        assert "sensitive_columns" in report
        assert "near_duplicates" in report
        assert "short_narratives" in report
        assert "total_samples" in report
        assert "assessment" in report

    def test_audit_counts_are_correct(self):
        data = self._sample_data()
        report = run_audit(data["narratives"], data["columns"])
        assert report["total_samples"] == 5
        assert report["redaction_markers"]["count"] > 0
        assert report["residual_pii"]["total"] >= 1  # john@test.com
        assert report["near_duplicates"]["count"] >= 1  # duplicate narrative

    def test_short_narratives_flagged(self):
        narratives = ["short", "This is a sufficiently long complaint narrative about fees"]
        report = run_audit(narratives, [])
        assert report["short_narratives"]["count"] >= 1

    def test_audit_writes_json(self):
        data = self._sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "audit.json"
            run_audit(data["narratives"], data["columns"], output_path=outpath)
            assert outpath.exists()
            loaded = json.loads(outpath.read_text())
            assert loaded["total_samples"] == 5


class TestPreTrainingGate:
    def test_gate_passes_when_under_threshold(self):
        narratives = ["Normal complaint about a billing issue"]
        report = run_audit(narratives, [], max_residual_pii=10)
        assert report["gate_passed"] is True

    def test_gate_fails_when_over_threshold(self):
        narratives = ["Contact john@a.com or jane@b.com or bob@c.com"]
        report = run_audit(narratives, [], max_residual_pii=1)
        assert report["gate_passed"] is False

    def test_gate_not_evaluated_when_no_threshold(self):
        narratives = ["Contact john@a.com"]
        report = run_audit(narratives, [])
        assert "gate_passed" not in report
