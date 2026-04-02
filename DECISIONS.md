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

## Why inverse-frequency class weights over resampling

CFPB product categories are heavily imbalanced (Debt collection dominates, Payday loan is rare). Inverse-frequency weights (`total / (num_classes × count_i)`) applied to `cross_entropy` give rare classes proportionally larger gradient signal without altering the training data distribution. The alternative — oversampling minority classes — would duplicate narratives and risk overfitting on rare-class phrasing. Weights are computed once from training labels and held constant; no dynamic curriculum.

## Why these corruption types and rates

The five corruptions test distinct failure modes a deployed model would face:

- **Typo injection** (10%, 20%): Simulates real-world OCR errors and user typos. Character-level operations (swap, delete, insert) with space protection. Two rates bracket mild vs severe degradation.
- **Token dropout** (20%, 40%): Tests positional dependence — whether the model relies on specific tokens or learns distributed representations. Special tokens ([CLS], [SEP]) are protected.
- **Truncation** (32 tokens): Simulates short-form input or API token limits. 32 tokens is aggressive — well below DistilBERT's 512 max — to surface hard degradation.
- **Tabular dropout** (50%): Tests graceful degradation when metadata is partially missing, common in production when upstream pipelines fail.
- **Tabular ablation** (100%): Measures how much the fusion head depends on metadata signal vs learning to ignore it.

Rates were chosen to produce a degradation spectrum — mild enough to be realistic at the low end, severe enough to differentiate architectures at the high end.

## Why exclusive modality dropout over independent

Modality dropout in M3 uses a single random draw per forward pass: 10% chance of zeroing the text branch, 10% chance of zeroing the tabular branch, 80% chance both survive. Independent dropout with the same per-branch rates would zero both modalities 1% of the time, producing a zero-information forward pass with no useful gradient. Exclusive dropout guarantees at least one modality is always active, forcing the model to learn complementary representations without wasting training steps on uninformative passes.

## Why macro-F1 as primary metric with 3-seed evaluation

Macro-F1 weights all classes equally regardless of prevalence, making it sensitive to performance on rare classes like Payday loan (which LightGBM scores 0.00 on). Accuracy alone would be misleading — a model predicting only Debt collection and Credit reporting could achieve >60% accuracy. Three seeds (42, 123, 456) provide mean and standard deviation to separate genuine architectural differences from initialization noise. The observed standard deviations (0.005–0.008) confirm that the +3.2pp fusion gain is well outside noise.

## Why the DatasetAdapter abstraction

The `DatasetAdapter` base class (`adapters/base.py`) defines a contract: `load_raw()` returns a DataFrame, `preprocess()` returns train/val/test splits with narratives, tabular features, labels, and company metadata. The CFPB adapter handles product merging, rare-class consolidation, leakage-safe feature encoding (top-50 companies, states, channels derived from training split only), and stratified splitting. A new dataset requires only implementing these two methods — the training loop, evaluation, and robustness pipeline are dataset-agnostic. This separates data engineering decisions from modeling decisions.

---

## Why AMP and gradient accumulation are disabled for DP-SGD runs

DP-SGD adds calibrated noise to per-sample gradients. AMP's dynamic loss scaling modifies gradient magnitudes in ways that interact unpredictably with Opacus's noise calibration and clipping. Disabling AMP for DP runs ensures the privacy accounting is correct — the epsilon value reported actually corresponds to the formal privacy guarantee. The ~2x training slowdown from disabling AMP is irrelevant on Modal A10G where each config completes in under an hour.

Gradient accumulation is also disabled because Opacus has its own `virtual_step_size` mechanism that is incompatible with manual accumulation.

## Why single learning rate for DP runs

The baseline uses differential learning rates (2e-5 encoder, 1e-3 head). Opacus's `PrivacyEngine` wraps the optimizer and may flatten param groups. Rather than adding compatibility shims, DP runs use a single 2e-5 learning rate. This is one more controlled variable between DP and non-DP runs — the experiment matrix documents what varies.

## Why loss-threshold MIA over shadow models

The loss-threshold attack is the simplest credible membership inference method: it requires only the trained model, no shadow models, no trained attack classifier. For this dataset size (20K samples) and purpose (empirical memorization measurement, not adversarial ML research), loss-threshold is sufficient. Shadow models would add significant compute cost and code complexity without changing the conclusion.

## Why balanced subsampling for MIA AUC validity

Unbalanced member/non-member sets inflate AUC in a way that's hard to compare across studies. We subsample training members to match the test set size, producing a 50/50 evaluation set. Both the subsample size and AUC are reported.

## Why exact normalized deduplication over MinHash

CFPB complaint narratives are short enough that near-duplicates are almost always exact duplicates with minor formatting differences. Exact match on normalized text (lowercased, whitespace-collapsed, punctuation-stripped) catches these with zero new dependencies and deterministic results. If MinHash for fuzzy near-duplicates is needed later, it's a follow-up, not a prerequisite.

## Why separate Modal image for privacy workloads

Opacus pulls specific PyTorch version constraints and its own build dependencies. `scripts/modal_privacy.py` uses a separate Modal app with its own image definition, keeping Opacus isolated from the main `scripts/modal_run.py` image. This prevents version contamination and build time increases for existing workloads.

## Why dual-scan data auditor over single PII scan

CFPB already redacts PII in published complaint narratives (replacing with XXXX, XX/XX/XXXX). A PII scanner that only reports raw counts on pre-redacted data would be misleading — low counts could suggest a broken scanner rather than effective source-level redaction. The dual-scan reports both redaction markers (documenting source controls) and residual PII (documenting what slipped through). This produces a more technically honest audit and surfaces the actually interesting finding: source redaction is imperfect.

## Why f-string template for model card, not Jinja2

The model card is ~200 lines of Markdown. A single `MODEL_CARD_TEMPLATE` constant with `{placeholders}` and one `.format()` call keeps the template readable and the data pipeline separate. Jinja2 would add a dependency for no gain at this complexity level. The template is a constant at the top of the file — easy to customize without touching data logic.

## Why model card sources from artifacts/ JSON, not MLflow

All model card inputs are serialized to `artifacts/` as JSON files (data audit, DP results, MIA results). The model card generator reads from one directory with no MLflow client dependency. MLflow is the source of truth during training; `artifacts/` is the source of truth for documentation. This decouples the documentation pipeline from the experiment tracking system.

## Why manual DP-SGD via vmap with per-group clipping

Six approaches were attempted before arriving at this solution. Each failure was diagnosed, not just observed.

**Approach 1: Raw Opacus on full DistilBERT.** Failed. Opacus 1.4's `GradSampleModule` cannot compute per-sample gradients through DistilBERT's multi-head attention and embedding layers. All three `grad_sample_mode` options (`hooks`, `ew`, `functorch`) fail with incompatible gradient tensor shapes. Tested on both Opacus 1.4.1 and 1.5.4 — same failure.

**Approach 2: dp-transformers (Microsoft).** Investigated and abandoned. The `convert_model_to_dp()` function referenced in the library's documentation does not exist in the released v1.0.0 package. The library provides only GPT-2 specific utilities (`convert_gpt2_attention_to_lora`, `force_causal_attention`). Additionally, dp-transformers 1.0.0 pins `torch<=1.12.1`, incompatible with our torch 2.2.2 stack.

**Approach 3: Frozen encoder + DP on head only (66K params).** Implemented and tested. F1 collapsed to 0.08 (random chance for 10 classes) across all epsilon values (1.0, 8.0, 50.0) with identical accuracy of 0.6685. The model learned to predict only the majority class. With only 66K trainable parameters, DP noise overwhelmed the gradient signal regardless of privacy budget.

**Approach 4: Opacus + LoRA adapters + head from scratch (~370K params).** The peft library's LoRA adapters inject vanilla `nn.Linear` layers into DistilBERT's attention projections (`q_lin`, `v_lin`), which Opacus handles natively. Spike test confirmed Opacus could wrap the model and complete a training step. However, full training produced the same F1=0.08 collapse. Root cause: **gradient heterogeneity under global per-sample clipping**. The fusion head's per-sample gradient norms (~5.0) dominate the global L2 norm. Opacus clips to `max_grad_norm=1.0`, so the clip factor is ~0.2, scaling all gradients down 5×. LoRA gradients (already ~0.01) become ~0.002, while noise remains at ~0.003/param. LoRA SNR drops below 0.5 — the adapters receive random walks instead of gradient signal. Tested across ε={0.5, 1.0, 8.0, 50.0}, LoRA rank={8, 16}, batch_size={16, 128, 256, 512}, epochs={3, 5, 10}, max_grad_norm={0.01, 0.1, 1.0}. All configurations produced identical F1=0.08. Reducing max_grad_norm doesn't help because global clipping scales signal and noise proportionally — the SNR ratio is invariant to the clip norm.

**Approach 5: Two-stage warm-start (non-DP head → DP LoRA fine-tune).** Trained LoRA + head without DP first (F1=0.54), then froze the head and DP-fine-tuned only the LoRA adapters. The DP stage preserved F1 (0.56 at ε=1.0, 0.56 at ε=50.0) — a flat curve. This is technically valid but meaningless: the head (which carries the classification signal) was trained without DP, so the privacy guarantee covers only the LoRA adapters, which the diagnostic showed contribute minimally. Resetting LoRA weights to random dropped F1 from 0.12 to 0.10 on the 2K test set — a real but small contribution.

**Approach 6 (current): Manual DP-SGD via `torch.func.vmap` with per-group clipping.** `vmap(grad(...))` computes per-sample gradients at the functional level, bypassing Opacus's `GradSampleModule` hooks entirely. This enables **per-group clipping**: LoRA parameters and head parameters are clipped with independent L2 norms (LoRA: C=0.1, head: C=1.0), with noise proportional to each group's clip norm. This directly solves the gradient heterogeneity problem — LoRA signal is not drowned by head gradient magnitude because they occupy separate clipping spaces. Opacus is used only for privacy accounting (`RDPAccountant` for budget tracking, `get_noise_multiplier` for noise calibration). Requires `attn_implementation='eager'` for DistilBERT because the default SDPA path contains `torch.all(mask == 1)` — data-dependent control flow that vmap cannot trace.

## Why T4 GPU over A10G for privacy workloads

The DP-SGD training loop with a frozen encoder is dominated by the forward pass through DistilBERT (feature extraction only, no per-sample gradients on the encoder). This is compute-light enough for T4 (16 GB VRAM, ~8 TFLOPS) — each of the 12 runs completes in under 15 minutes. A10G (24 GB, ~31 TFLOPS) would be faster but costs ~3× more per hour. Since the runs parallelize via Modal starmap, wall-clock time is bounded by the slowest run regardless of GPU tier. T4 is the right cost-performance point for this workload.

## Why model cards as governance artifact

Model cards (Mitchell et al. 2019) are the industry-standard documentation format for ML model governance and are increasingly referenced by the EU AI Act's transparency requirements. The model card auto-generates from JSON artifacts so it stays current with the training pipeline rather than becoming stale documentation. All 8 Mitchell et al. sections are populated: model details, intended use, training data, evaluation results, fairness analysis, privacy, limitations & risks, and deployment recommendations.

## Why NER entities are reported but not used as a PII gate

The data audit reports 114,758 NER entities (59,799 ORG, 36,527 PERSON, 18,365 GPE) but the PII gate only enforces on regex-detected residual PII (emails, phones, SSNs). NER entities in CFPB data are expected content, not PII leakage — company names, complainant roles, states, and cities are the actual features the model classifies on. Gating on ORG entities would block training on a complaint classification dataset where company identity is a primary signal. The NER scan exists for transparency: it documents what entity types are present so downstream consumers can make informed decisions. The regex gate exists for safety: it catches contact information that survived CFPB's source-level redaction. These are different concerns with different thresholds.
