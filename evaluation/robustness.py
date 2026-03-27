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
    attention_mask: torch.Tensor | None = None,
    rate: float = 0.2,
    pad_id: int = 0,
    cls_id: int = 101,
    sep_id: int = 102,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Replace tokens with [PAD] and zero their attention_mask, preserving [CLS] and [SEP].

    Returns (modified_input_ids, modified_attention_mask).
    """
    if rate == 0.0:
        return input_ids.clone(), attention_mask.clone() if attention_mask is not None else None

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    result = input_ids.clone()
    mask = (torch.rand(result.shape, generator=gen) < rate).to(result.device)

    # Protect special tokens
    mask = mask & (result != cls_id) & (result != sep_id) & (result != pad_id)

    result[mask] = pad_id

    result_mask = None
    if attention_mask is not None:
        result_mask = attention_mask.clone()
        result_mask[mask] = 0

    return result, result_mask


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

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    mask = (torch.rand(features.shape, generator=gen) >= rate).to(features.device)
    return features * mask.float()


def tabular_ablation(features: torch.Tensor) -> torch.Tensor:
    """Zero out all tabular features (simulate fully missing metadata)."""
    return torch.zeros_like(features)


def run_robustness_eval(
    model,
    test_narratives: list[str],
    test_tabular,
    test_labels: list[int],
    class_names: list[str],
    tokenizer,
    max_length: int = 128,
    device: torch.device = None,
    seed: int = 42,
    is_text_only: bool = False,
) -> dict[str, dict]:
    """Run full robustness evaluation suite.

    Args:
        test_tabular: Tabular features as np.ndarray or torch.Tensor.
        is_text_only: If True, skip tabular corruption entries (N/A for M1).

    Returns dict mapping corruption_name -> {macro_f1, accuracy}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np

    from evaluation.metrics import compute_metrics

    # Convert np.ndarray from adapter to tensor
    if isinstance(test_tabular, np.ndarray):
        test_tabular = torch.tensor(test_tabular, dtype=torch.float32)

    corruptions = [
        ("clean", {}),
        ("typo_10", {"type": "typo", "rate": 0.1}),
        ("typo_20", {"type": "typo", "rate": 0.2}),
        ("token_drop_20", {"type": "token_drop", "rate": 0.2}),
        ("token_drop_40", {"type": "token_drop", "rate": 0.4}),
        ("truncate_32", {"type": "truncate", "max_tokens": 32}),
        ("tabular_drop_50", {"type": "tabular_drop", "rate": 0.5}),
        ("tabular_ablation", {"type": "tabular_ablation"}),
    ]

    model.eval()
    results = {}

    for name, params in corruptions:
        corruption_type = params.get("type")

        # Skip tabular corruptions for text-only models
        if is_text_only and corruption_type in ("tabular_drop", "tabular_ablation"):
            results[name] = None  # Signals N/A in table generation
            continue

        all_preds = []

        # Apply text-level corruptions
        if corruption_type == "typo":
            corrupted_texts = [
                inject_typos(t, rate=params["rate"], seed=seed + i)
                for i, t in enumerate(test_narratives)
            ]
        elif corruption_type == "truncate":
            corrupted_texts = [
                truncate_text(t, max_tokens=params["max_tokens"])
                for t in test_narratives
            ]
        else:
            corrupted_texts = test_narratives

        batch_size = 16
        for start in range(0, len(corrupted_texts), batch_size):
            end = min(start + batch_size, len(corrupted_texts))
            batch_texts = corrupted_texts[start:end]
            batch_tabular = test_tabular[start:end].to(device)

            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in encodings.items()}

            if corruption_type == "token_drop":
                text_inputs["input_ids"], text_inputs["attention_mask"] = token_dropout(
                    text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    rate=params["rate"],
                    seed=seed + start,
                )

            if corruption_type == "tabular_drop":
                batch_tabular = tabular_dropout(
                    batch_tabular, rate=params["rate"], seed=seed + start,
                )
            elif corruption_type == "tabular_ablation":
                batch_tabular = tabular_ablation(batch_tabular)

            with torch.no_grad():
                logits = model(text_inputs, batch_tabular)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

        metrics = compute_metrics(test_labels, all_preds, class_names)
        results[name] = {
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
        }
        print(f"  {name}: F1={metrics.macro_f1:.4f}, Acc={metrics.accuracy:.4f}")

    return results
