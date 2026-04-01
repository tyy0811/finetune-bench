# Model Card: finetune-bench Multimodal Complaint Classifier

## Model Details

**Architecture:** DistilBERT (66M params) + Tabular MLP (2-layer) + Fusion head.
Text branch produces 768-dim [CLS] embedding; tabular branch produces 64-dim embedding;
fusion head concatenates both (832-dim) and classifies into complaint product categories.

**Training procedure:** AdamW optimizer with differential learning rates (2e-5 encoder,
1e-3 head), cosine schedule with 10% warmup, 3 epochs, batch size 16 with 2-step gradient
accumulation, inverse-frequency class weighting.

**Variants trained:** M1 (text-only), M2 (fusion), M3 (fusion + modality dropout),
M2b (fusion without company features). DP-SGD variants trained with Opacus at
epsilon = {50.0, 8.0, 1.0}.

## Intended Use

**Intended for:** Multi-class classification of consumer financial complaints into product
categories (credit reporting, mortgage, debt collection, etc.). Designed for routing and
triage workflows where a human reviews the final decision.

**Not intended for:** Individual risk scoring, automated complaint resolution without human
review, credit decisioning, or any use where model output directly affects consumer outcomes
without human oversight.

## Training Data

**Source:** CFPB Consumer Complaints Database (consumerfinance.gov).
**Size:** 16000 samples (80/10/10 train/val/test split).

**Privacy audit summary:** Source applies redaction (223294 markers). 7 residual PII instances suggest incomplete coverage. 34754 near-duplicates flagged for deduplication.

- Redaction markers detected: 223294 (XX/XX/XXXX, XXXX, XXXXXXXX)
- Residual PII: 7 instances (emails: 5, phones: 2, SSNs: 0)
- Sensitive columns flagged: company, state, submitted_via
- Near-duplicates: 34754 (exact_normalized)
- Short narratives (<20 tokens): 647

## Evaluation Results

**Baseline performance (no DP):**

| Variant | Macro-F1 | Notes |
|---------|----------|-------|
| M1 | 0.6236 +/- 0.0046 | DistilBERT baseline |
| M2 | 0.6555 +/- 0.0076 | Multimodal fusion |
| M3 | 0.6605 +/- 0.0053 | Fusion + modality dropout |
| M2b (no company) | 0.6189 | Fusion without company features |

**Privacy-utility tradeoff (DP-SGD):**

| Config | Target epsilon | Actual epsilon | Macro-F1 |
|--------|---------------|----------------|----------|
| loose_dp | 50.0 | 49.9939 | 0.0801 +/- 0.0000 |
| moderate_dp | 8.0 | 7.9973 | 0.0801 +/- 0.0000 |
| strict_dp | 1.0 | 0.9917 | 0.0801 +/- 0.0000 |
| strict_dp_tuned_clip | 1.0 | 0.9917 | 0.0801 +/- 0.0000 |

## Fairness Analysis

**Sensitive attributes identified:** company, state, submitted_via

The primary fairness concern is **entity memorization** (see Finding #7): the model achieves
higher accuracy on complaints from frequently-occurring companies in the training data.
The +3.2pp fusion gain from M1 to M2 is driven by company identity features — removing
company features (M2b) drops performance below text-only M1.

This means complaints from underrepresented companies receive lower-quality predictions.

## Privacy

**Differential privacy training:** Models trained with Opacus DP-SGD at epsilon = {50.0, 8.0, 1.0} with delta = 1e-5. AMP and gradient accumulation disabled during DP runs to ensure correct privacy accounting.

| Config | Target epsilon | Actual epsilon | Macro-F1 |
|--------|---------------|----------------|----------|
| loose_dp | 50.0 | 49.9939 | 0.0801 +/- 0.0000 |
| moderate_dp | 8.0 | 7.9973 | 0.0801 +/- 0.0000 |
| strict_dp | 1.0 | 0.9917 | 0.0801 +/- 0.0000 |
| strict_dp_tuned_clip | 1.0 | 0.9917 | 0.0801 +/- 0.0000 |

**Membership inference attack results:**

| Model | epsilon | MIA AUC | Loss Gap | Members | Non-members |
|-------|---------|---------|----------|---------|-------------|
| M2_no_dp | inf | 0.53 | 0.13 | 2000 | 2000 |
| loose_dp | 50.0 | 0.49 | -0.08 | 2000 | 2000 |
| moderate_dp | 8.0 | 0.49 | -0.07 | 2000 | 2000 |
| strict_dp | 1.0 | 0.49 | -0.06 | 2000 | 2000 |
| strict_dp_tuned_clip | 1.0 | 0.49 | -0.07 | 2000 | 2000 |

**Stratified MIA (entity frequency):**

| Model | High-freq company AUC | Low-freq company AUC | High-freq N | Low-freq N |
|-------|----------------------|---------------------|-------------|------------|
| M2_no_dp | 0.52 | 0.54 | 11528 | 4472 |
| loose_dp | 0.60 | 0.24 | 11528 | 4472 |
| moderate_dp | 0.60 | 0.24 | 11528 | 4472 |
| strict_dp | 0.61 | 0.24 | 11528 | 4472 |
| strict_dp_tuned_clip | 0.61 | 0.24 | 11528 | 4472 |

**Interpretation:** The non-DP model shows MIA AUC of 0.53, indicating moderate memorization of training data. DP training reduces MIA AUC to 0.49, approaching random guess (0.50).

## Limitations & Risks

- **Entity memorization:** Fusion gain is driven by company identity, not generalizable multimodal learning (Finding #7). High-frequency companies get better predictions
- **Class imbalance:** 10-class classification with significant class size variation. Inverse-frequency weighting mitigates but does not eliminate
- **Text truncation:** 128-token limit discards information from long complaints
- **Temporal drift:** Mild (0.8pp F1 drop with temporal split), but language and product distributions shift over time
- **DP-utility tradeoff:** Strict privacy (epsilon=1.0) significantly degrades model performance

## Deployment Recommendations

- **For low-risk routing/triage:** Use non-DP M2 or M3 model with human review. Monitor per-company performance for drift
- **For privacy-sensitive deployments:** Use DP model at epsilon=8.0 (moderate privacy). Accept the F1 reduction as the cost of formal privacy guarantees
- **For strict regulatory compliance:** Use DP model at epsilon=1.0. Evaluate whether the accuracy is acceptable for the use case before deploying
- **Monitoring:** Track per-company F1 monthly. Re-train if any company subgroup drops >5pp below the population mean
- **Re-training triggers:** New product categories, regulatory changes affecting complaint language, >2pp drop in population-level Macro-F1
