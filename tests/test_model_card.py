"""Tests for scripts/generate_model_card.py."""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_model_card import MODEL_CARD_TEMPLATE, generate_model_card

FIXTURES = Path(__file__).parent / "fixtures"


class TestModelCardTemplate:
    def test_template_has_all_sections(self):
        """The template string must define all 8 Mitchell et al. sections."""
        required_sections = [
            "Model Details",
            "Intended Use",
            "Training Data",
            "Evaluation Results",
            "Fairness Analysis",
            "Privacy",
            "Limitations & Risks",
            "Deployment Recommendations",
        ]
        for section in required_sections:
            assert f"## {section}" in MODEL_CARD_TEMPLATE, f"Missing section: {section}"


class TestGenerateModelCard:
    def _load_fixtures(self):
        return {
            "audit": json.loads((FIXTURES / "data_audit_report.json").read_text()),
            "dp": json.loads((FIXTURES / "dp_results.json").read_text()),
            "mia": json.loads((FIXTURES / "mia_results.json").read_text()),
        }

    def test_generates_markdown(self):
        data = self._load_fixtures()
        card = generate_model_card(data["audit"], data["dp"], data["mia"])
        assert isinstance(card, str)
        assert len(card) > 100

    def test_all_sections_present(self):
        data = self._load_fixtures()
        card = generate_model_card(data["audit"], data["dp"], data["mia"])
        required = [
            "Model Details", "Intended Use", "Training Data",
            "Evaluation Results", "Fairness Analysis", "Privacy",
            "Limitations & Risks", "Deployment Recommendations",
        ]
        for section in required:
            assert f"## {section}" in card, f"Missing section: {section}"

    def test_no_empty_sections(self):
        data = self._load_fixtures()
        card = generate_model_card(data["audit"], data["dp"], data["mia"])
        # An empty section would be ## Heading\n\n## Next Heading
        empty_pattern = re.compile(r"## .+\n\n## ")
        assert not empty_pattern.search(card), "Found empty section in model card"

    def test_numeric_consistency(self):
        data = self._load_fixtures()
        card = generate_model_card(data["audit"], data["dp"], data["mia"])
        # Audit numbers should appear in card
        assert "3200" in card  # redaction marker count
        assert "0.73" in card  # MIA AUC for no-DP model

    def test_writes_to_file(self):
        data = self._load_fixtures()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "model_card.md"
            generate_model_card(
                data["audit"], data["dp"], data["mia"], output_path=outpath
            )
            assert outpath.exists()
            content = outpath.read_text()
            assert "## Model Details" in content
