"""Tests for corruption functions."""

import torch

from evaluation.robustness import (
    inject_typos,
    tabular_ablation,
    tabular_dropout,
    token_dropout,
    truncate_text,
)


class TestInjectTypos:
    def test_returns_string(self):
        result = inject_typos("hello world", rate=0.5, seed=42)
        assert isinstance(result, str)

    def test_rate_zero_unchanged(self):
        text = "hello world"
        assert inject_typos(text, rate=0.0, seed=42) == text

    def test_rate_one_changes_all_non_space(self):
        text = "hello world"
        result = inject_typos(text, rate=1.0, seed=42)
        assert result != text

    def test_deterministic_with_seed(self):
        text = "the quick brown fox"
        r1 = inject_typos(text, rate=0.3, seed=42)
        r2 = inject_typos(text, rate=0.3, seed=42)
        assert r1 == r2

    def test_swap_actually_swaps(self):
        """Verify swap produces transposition, not duplication."""
        text = "ab"
        result = inject_typos(text, rate=1.0, seed=0)
        assert result != "bb", "Swap should transpose, not duplicate"


class TestTokenDropout:
    def test_shape_preserved(self):
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        result_ids, _ = token_dropout(input_ids, rate=0.5, seed=42)
        assert result_ids.shape == input_ids.shape

    def test_cls_sep_preserved(self):
        """[CLS]=101, [SEP]=102 should never be dropped."""
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        result_ids, _ = token_dropout(input_ids, rate=1.0, seed=42)
        assert result_ids[0, 0].item() == 101  # [CLS]
        # Find [SEP] position (not necessarily last due to padding)
        sep_positions = (input_ids == 102).nonzero(as_tuple=True)
        for pos in sep_positions[1]:
            assert result_ids[0, pos.item()].item() == 102

    def test_rate_zero_unchanged(self):
        input_ids = torch.tensor([[101, 2023, 2003, 102]])
        result_ids, _ = token_dropout(input_ids, rate=0.0, seed=42)
        assert torch.equal(result_ids, input_ids)

    def test_attention_mask_zeroed(self):
        """Dropped tokens must also have their attention_mask zeroed."""
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        attn_mask = torch.ones_like(input_ids)
        result_ids, result_mask = token_dropout(
            input_ids, attention_mask=attn_mask, rate=1.0, seed=42,
        )
        # Wherever input_ids became pad_id (0), attention_mask must be 0
        dropped = (result_ids == 0) & (input_ids != 0)
        assert torch.all(result_mask[dropped] == 0)


class TestTruncateText:
    def test_truncation(self):
        text = "one two three four five six seven eight"
        result = truncate_text(text, max_tokens=3)
        assert result == "one two three"

    def test_short_text_unchanged(self):
        text = "hello"
        result = truncate_text(text, max_tokens=10)
        assert result == text


class TestTabularDropout:
    def test_shape_preserved(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=0.5, seed=42)
        assert result.shape == features.shape

    def test_rate_zero_unchanged(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=0.0, seed=42)
        assert torch.equal(result, features)

    def test_rate_one_all_zero(self):
        features = torch.randn(4, 10)
        result = tabular_dropout(features, rate=1.0, seed=42)
        assert torch.all(result == 0)


class TestTabularAblation:
    def test_all_zeros(self):
        features = torch.randn(4, 10)
        result = tabular_ablation(features)
        assert torch.all(result == 0)

    def test_shape_preserved(self):
        features = torch.randn(4, 10)
        result = tabular_ablation(features)
        assert result.shape == features.shape
