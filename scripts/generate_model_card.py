"""Generate a model card from training artifacts.

Reads JSON files from artifacts/ and produces artifacts/model_card.md.
Template structure follows Mitchell et al. 2019 model card framework.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_CARD_TEMPLATE = """\
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
epsilon = {{50.0, 8.0, 1.0}}.

## Intended Use

**Intended for:** Multi-class classification of consumer financial complaints into product
categories (credit reporting, mortgage, debt collection, etc.). Designed for routing and
triage workflows where a human reviews the final decision.

**Not intended for:** Individual risk scoring, automated complaint resolution without human
review, credit decisioning, or any use where model output directly affects consumer outcomes
without human oversight.

## Training Data

**Source:** CFPB Consumer Complaints Database (consumerfinance.gov).
**Size:** {total_samples} samples (80/10/10 train/val/test split).

**Privacy audit summary:** {audit_assessment}

- Redaction markers detected: {redaction_count} ({redaction_patterns})
- Residual PII: {residual_pii_total} instances (emails: {residual_pii_emails}, \
phones: {residual_pii_phones}, SSNs: {residual_pii_ssns})
- Sensitive columns flagged: {sensitive_columns}
- Near-duplicates: {near_duplicate_count} ({near_duplicate_method})
- Short narratives (<{short_threshold} tokens): {short_count}

## Evaluation Results

**Baseline performance (no DP):**

| Variant | Macro-F1 | Notes |
|---------|----------|-------|
| M1 (text-only) | 0.6236 +/- 0.0047 | DistilBERT baseline |
| M2 (fusion) | 0.6555 +/- 0.0076 | +3.2pp from fusion |
| M3 (fusion+dropout) | 0.6605 +/- 0.0053 | +0.5pp from modality dropout |
| M2b (no company) | 0.6189 | Fusion gain disappears without company features |

**Privacy-utility tradeoff (DP-SGD):**

{dp_results_table}

## Fairness Analysis

**Sensitive attributes identified:** {sensitive_columns}

The primary fairness concern is **entity memorization** (see Finding #7): the model achieves
higher accuracy on complaints from frequently-occurring companies in the training data.
The +3.2pp fusion gain from M1 to M2 is driven by company identity features — removing
company features (M2b) drops performance below text-only M1.

This means complaints from underrepresented companies receive lower-quality predictions.

## Privacy

**Differential privacy training:** Models trained with Opacus DP-SGD at epsilon = \
{{50.0, 8.0, 1.0}} with delta = 1e-5. AMP and gradient accumulation disabled during DP \
runs to ensure correct privacy accounting.

{dp_results_table_privacy}

**Membership inference attack results:**

{mia_results_table}

**Stratified MIA (entity frequency):**

{stratified_mia_table}

**Interpretation:** {mia_interpretation}

## Limitations & Risks

- **Entity memorization:** Fusion gain is driven by company identity, not generalizable \
multimodal learning (Finding #7). High-frequency companies get better predictions
- **Class imbalance:** 10-class classification with significant class size variation. \
Inverse-frequency weighting mitigates but does not eliminate
- **Text truncation:** 128-token limit discards information from long complaints
- **Temporal drift:** Mild (0.8pp F1 drop with temporal split), but language and product \
distributions shift over time
- **DP-utility tradeoff:** Strict privacy (epsilon=1.0) significantly degrades model \
performance

## Deployment Recommendations

- **For low-risk routing/triage:** Use non-DP M2 or M3 model with human review. Monitor \
per-company performance for drift
- **For privacy-sensitive deployments:** Use DP model at epsilon=8.0 (moderate privacy). \
Accept the F1 reduction as the cost of formal privacy guarantees
- **For strict regulatory compliance:** Use DP model at epsilon=1.0. Evaluate whether the \
accuracy is acceptable for the use case before deploying
- **Monitoring:** Track per-company F1 monthly. Re-train if any company subgroup drops \
>5pp below the population mean
- **Re-training triggers:** New product categories, regulatory changes affecting complaint \
language, >2pp drop in population-level Macro-F1
"""


def _format_dp_table(dp_data: dict) -> str:
    """Format DP results as a Markdown table."""
    rows = [
        "| Config | Target epsilon | Actual epsilon | Macro-F1 |",
        "|--------|---------------|----------------|----------|",
    ]
    for r in dp_data["results"]:
        rows.append(
            f"| {r['config']} | {r['epsilon_target']} | {r['epsilon_actual']} "
            f"| {r['val_macro_f1']:.4f} +/- {r['val_macro_f1_std']:.4f} |"
        )
    return "\n".join(rows)


def _format_mia_table(mia_data: dict) -> str:
    """Format MIA results as a Markdown table."""
    rows = [
        "| Model | epsilon | MIA AUC | Loss Gap | Members | Non-members |",
        "|-------|---------|---------|----------|---------|-------------|",
    ]
    for r in mia_data["results"]:
        rows.append(
            f"| {r['model']} | {r['epsilon']} | {r['mia_auc']:.2f} "
            f"| {r['loss_gap']:.2f} | {r['member_sample_size']} "
            f"| {r['non_member_sample_size']} |"
        )
    return "\n".join(rows)


def _format_stratified_mia_table(mia_data: dict) -> str:
    """Format stratified MIA results as a Markdown table."""
    rows = [
        "| Model | High-freq company AUC | Low-freq company AUC "
        "| High-freq N | Low-freq N |",
        "|-------|----------------------|---------------------"
        "|-------------|------------|",
    ]
    for r in mia_data["results"]:
        s = r.get("stratified", {})
        hf_auc = s.get("high_freq_company_auc", "N/A")
        lf_auc = s.get("low_freq_company_auc", "N/A")
        if isinstance(hf_auc, float):
            hf_auc = f"{hf_auc:.2f}"
        if isinstance(lf_auc, float):
            lf_auc = f"{lf_auc:.2f}"
        rows.append(
            f"| {r['model']} | {hf_auc} | {lf_auc} "
            f"| {s.get('high_freq_count', 'N/A')} | {s.get('low_freq_count', 'N/A')} |"
        )
    return "\n".join(rows)


def _mia_interpretation(mia_data: dict) -> str:
    """Generate a one-paragraph interpretation of MIA results."""
    results = mia_data["results"]
    if len(results) < 2:
        return "Insufficient model variants for comparative analysis."

    no_dp = [r for r in results if r["epsilon"] in ("inf", float("inf"))]
    dp = [r for r in results if r["epsilon"] not in ("inf", float("inf"))]

    parts = []
    if no_dp:
        parts.append(
            f"The non-DP model shows MIA AUC of {no_dp[0]['mia_auc']:.2f}, "
            f"indicating {'significant' if no_dp[0]['mia_auc'] > 0.65 else 'moderate'} "
            f"memorization of training data."
        )
    if dp:
        lowest_auc = min(r["mia_auc"] for r in dp)
        parts.append(
            f"DP training reduces MIA AUC to {lowest_auc:.2f}, "
            f"{'approaching random guess (0.50)' if lowest_auc < 0.55 else 'showing reduced but non-trivial memorization'}."
        )

    # Check stratified results
    if no_dp and no_dp[0].get("stratified"):
        s = no_dp[0]["stratified"]
        if s.get("high_freq_company_auc") and s.get("low_freq_company_auc"):
            gap = s["high_freq_company_auc"] - s["low_freq_company_auc"]
            if gap > 0.05:
                parts.append(
                    f"Stratified analysis confirms entity memorization: "
                    f"high-frequency companies show {gap:.2f} higher MIA AUC than "
                    f"low-frequency companies, consistent with Finding #7."
                )

    return " ".join(parts)


def generate_model_card(
    audit_data: dict,
    dp_data: dict,
    mia_data: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a model card from audit, DP, and MIA results."""
    card = MODEL_CARD_TEMPLATE.format(
        total_samples=audit_data["total_samples"],
        audit_assessment=audit_data["assessment"],
        redaction_count=audit_data["redaction_markers"]["count"],
        redaction_patterns=", ".join(audit_data["redaction_markers"]["patterns_found"]),
        residual_pii_total=audit_data["residual_pii"]["total"],
        residual_pii_emails=audit_data["residual_pii"]["emails"],
        residual_pii_phones=audit_data["residual_pii"]["phones"],
        residual_pii_ssns=audit_data["residual_pii"]["ssns"],
        sensitive_columns=", ".join(audit_data["sensitive_columns"]),
        near_duplicate_count=audit_data["near_duplicates"]["count"],
        near_duplicate_method=audit_data["near_duplicates"]["method"],
        short_threshold=audit_data["short_narratives"]["threshold_tokens"],
        short_count=audit_data["short_narratives"]["count"],
        dp_results_table=_format_dp_table(dp_data),
        dp_results_table_privacy=_format_dp_table(dp_data),
        mia_results_table=_format_mia_table(mia_data),
        stratified_mia_table=_format_stratified_mia_table(mia_data),
        mia_interpretation=_mia_interpretation(mia_data),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card)

    return card


def main() -> None:
    """Generate model card from artifacts/ directory."""
    artifacts = Path("artifacts")

    audit = json.loads((artifacts / "data_audit_report.json").read_text())
    dp = json.loads((artifacts / "dp_results.json").read_text())
    mia = json.loads((artifacts / "mia_results.json").read_text())

    output = artifacts / "model_card.md"
    generate_model_card(audit, dp, mia, output_path=output)
    print(f"Model card written to {output}")


if __name__ == "__main__":
    main()
