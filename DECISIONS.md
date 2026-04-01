# Design Decisions

## Why manual AMP integration, not HuggingFace Trainer

The training loop is custom (AdamW + linear warmup + cosine decay + gradient
accumulation + modality dropout + differential learning rates). HuggingFace
Trainer would abstract away the details that make this a portfolio piece.

Manual `torch.cuda.amp` integration with `GradScaler` demonstrates
understanding of the underlying mechanism: which operations run in fp16
(matmul, convolutions), which stay in fp32 (softmax, layer norm, loss), and
how loss scaling prevents gradient underflow in fp16.

## Why fp16 with GradScaler, not bf16

fp16 with GradScaler works on all CUDA GPUs (Volta and newer) and
demonstrates the full loss-scaling mechanism. bf16 (available on Ampere+) has
a larger dynamic range that eliminates the need for loss scaling — simpler in
production but less instructive as a portfolio piece.

**Production recommendation:** On Ampere+ GPUs (A10G, A100, H100), use bf16
via `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` with no
GradScaler. Simpler code, no risk of gradient underflow.

## Why honest fp16 limitations reporting

DistilBERT has 66M parameters. On an NVIDIA A10 (24 GB), fp16 reduces peak
memory by only 8% (1952 MB -> 1797 MB) — not the dramatic 50%+ savings seen
on 7B+ parameter models. The real win is a 2x epoch speedup from Tensor Core
acceleration (70.9s -> 36.2s for M2). Macro-F1 is preserved within noise
(0.6562 fp32 vs 0.6570 fp16). Reporting this honestly — including the GPU
name, total VRAM, and utilization percentage — shows engineering maturity.
Inflating the benefit would be caught by any reviewer who understands the
relationship between model size and precision scaling.

---

## Design Decision #4: AMP and gradient accumulation disabled for DP-SGD runs

DP-SGD adds calibrated noise to per-sample gradients. AMP's dynamic loss scaling modifies gradient magnitudes in ways that interact unpredictably with Opacus's noise calibration and clipping. Disabling AMP for DP runs ensures the privacy accounting is correct — the epsilon value reported actually corresponds to the formal privacy guarantee. The ~2x training slowdown from disabling AMP is irrelevant on Modal A10G where each config completes in under an hour.

Gradient accumulation is also disabled because Opacus has its own `virtual_step_size` mechanism that is incompatible with manual accumulation.

## Design Decision #5: Single learning rate for DP runs

The baseline uses differential learning rates (2e-5 encoder, 1e-3 head). Opacus's `PrivacyEngine` wraps the optimizer and may flatten param groups. Rather than adding compatibility shims, DP runs use a single 2e-5 learning rate. This is one more controlled variable between DP and non-DP runs — the experiment matrix documents what varies.

## Design Decision #6: Loss-threshold MIA over shadow models

The loss-threshold attack is the simplest credible membership inference method: it requires only the trained model, no shadow models, no trained attack classifier. For this dataset size (20K samples) and purpose (empirical memorization measurement, not adversarial ML research), loss-threshold is sufficient. Shadow models would add significant compute cost and code complexity without changing the conclusion.

## Design Decision #7: Balanced subsampling for MIA AUC validity

Unbalanced member/non-member sets inflate AUC in a way that's hard to compare across studies. We subsample training members to match the test set size, producing a 50/50 evaluation set. Both the subsample size and AUC are reported.

## Design Decision #8: Exact normalized deduplication over MinHash

CFPB complaint narratives are short enough that near-duplicates are almost always exact duplicates with minor formatting differences. Exact match on normalized text (lowercased, whitespace-collapsed, punctuation-stripped) catches these with zero new dependencies and deterministic results. If MinHash for fuzzy near-duplicates is needed later, it's a follow-up, not a prerequisite.

## Design Decision #9: Separate Modal image for privacy workloads

Opacus pulls specific PyTorch version constraints and its own build dependencies. `scripts/modal_privacy.py` uses a separate Modal app with its own image definition, keeping Opacus isolated from the main `scripts/modal_run.py` image. This prevents version contamination and build time increases for existing workloads.

## Design Decision #10: Dual-scan data auditor (redaction markers + residual PII)

CFPB already redacts PII in published complaint narratives (replacing with XXXX, XX/XX/XXXX). A PII scanner that only reports raw counts on pre-redacted data would be misleading — low counts could suggest a broken scanner rather than effective source-level redaction. The dual-scan reports both redaction markers (documenting source controls) and residual PII (documenting what slipped through). This produces a more technically honest audit and surfaces the actually interesting finding: source redaction is imperfect.

## Design Decision #11: f-string template for model card, not Jinja2

The model card is ~200 lines of Markdown. A single `MODEL_CARD_TEMPLATE` constant with `{placeholders}` and one `.format()` call keeps the template readable and the data pipeline separate. Jinja2 would add a dependency for no gain at this complexity level. The template is a constant at the top of the file — easy to customize without touching data logic.

## Design Decision #12: Model card sources from artifacts/ JSON, not MLflow

All model card inputs are serialized to `artifacts/` as JSON files (data audit, DP results, MIA results). The model card generator reads from one directory with no MLflow client dependency. MLflow is the source of truth during training; `artifacts/` is the source of truth for documentation. This decouples the documentation pipeline from the experiment tracking system.

## Design Decision #13: Frozen DistilBERT encoder for DP-SGD training

Opacus 1.4 cannot compute per-sample gradients through DistilBERT's multi-head attention and LayerNorm layers — the backward hooks produce inconsistent gradient shapes across parameters. Freezing the encoder and DP-training only the tabular MLP + fusion head (66K trainable params) is the standard workaround documented by the Opacus team for transformer architectures. The privacy guarantee applies to the trained parameters; the frozen encoder acts as a fixed feature extractor. This is a meaningful limitation: the DP model cannot adapt the text representations, so its utility ceiling is lower than the non-DP baseline which fine-tunes the full model.
