"""Robustness corruption functions for evaluation.

Text corruptions: applied to raw text before tokenization.
Token dropout: operates on tokenized input_ids.
Tabular corruptions: operate on feature tensors.
"""

import random
import string

import torch


def inject_typos(text: str, rate: float = 0.1, seed: int | None = None) -> str:
    """Inject character-level typos at the given rate.

    For each character, with probability=rate, apply one of:
    - swap with adjacent character (true transposition)
    - delete the character
    - insert a random lowercase letter

    Spaces are never corrupted.
    """
    if rate == 0.0:
        return text

    rng = random.Random(seed)
    chars = list(text)
    i = 0
    result = []

    while i < len(chars):
        ch = chars[i]

        if ch == " " or rng.random() >= rate:
            result.append(ch)
            i += 1
            continue

        op = rng.choice(["swap", "delete", "insert"])

        if op == "swap" and i + 1 < len(chars) and chars[i + 1] != " ":
            # True transposition: emit chars[i+1] then chars[i], skip both
            result.append(chars[i + 1])
            result.append(chars[i])
            i += 2  # skip the next character since we already emitted it
        elif op == "delete":
            i += 1  # skip this character
        elif op == "insert":
            result.append(ch)
            result.append(rng.choice(string.ascii_lowercase))
            i += 1
        else:
            # swap at end of string — fall back to insert
            result.append(ch)
            result.append(rng.choice(string.ascii_lowercase))
            i += 1

    return "".join(result)


def token_dropout(
    input_ids: torch.Tensor,
    rate: float = 0.2,
    pad_id: int = 0,
    cls_id: int = 101,
    sep_id: int = 102,
    seed: int | None = None,
) -> torch.Tensor:
    """Replace tokens with [PAD], preserving [CLS] and [SEP]."""
    if rate == 0.0:
        return input_ids.clone()

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    result = input_ids.clone()
    mask = torch.rand(result.shape, generator=gen) < rate

    # Protect special tokens
    mask = mask & (result != cls_id) & (result != sep_id) & (result != pad_id)

    result[mask] = pad_id
    return result


def truncate_text(text: str, max_tokens: int = 32) -> str:
    """Keep only the first N whitespace-split tokens."""
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def tabular_dropout(
    features: torch.Tensor,
    rate: float = 0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Zero out each feature independently with probability=rate."""
    if rate == 0.0:
        return features.clone()
    if rate == 1.0:
        return torch.zeros_like(features)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    mask = torch.rand(features.shape, generator=gen) >= rate
    return features * mask.float()


def tabular_ablation(features: torch.Tensor) -> torch.Tensor:
    """Zero out all tabular features (simulate fully missing metadata)."""
    return torch.zeros_like(features)
