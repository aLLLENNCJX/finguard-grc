"""Automated Evaluation Report Generator (docs/evaluation_report.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finguard.config import DOCS_DIR


def build_evaluation_report(
    benchmark_results: dict[str, Any],
    rag_eval: dict[str, Any],
    domain_maturity: list[dict[str, Any]],
    output_path: Path = DOCS_DIR / "evaluation_report.md",
) -> Path:
    """Write reproducible markdown evaluation report using actual computed metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split = benchmark_results["split_info"]
    rows = benchmark_results["benchmarks"]
    err = benchmark_results["error_analysis"]
    champ = rows[-1]

    table_lines = [
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | F2 | FPR | Alert Compression | Hrs Saved / Qtr |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        table_lines.append(
            f"| {r['model_name']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['f2']:.4f} | "
            f"{r['fpr']:.4f} | {r['alert_compression_ratio']*100:.1f}% | {r['analyst_hours_saved_quarter']:.1f}h |"
        )

    ablation_lines = [
        "| Retrieval Mode | Recall@1 | Recall@3 | MRR |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for ab in rag_eval.get("ablation", []):
        ablation_lines.append(
            f"| {ab['mode']} | {ab['recall_at_1']*100:.1f}% | {ab['recall_at_3']*100:.1f}% | {ab['mrr']:.4f} |"
        )

    maturity_lines = [
        "| Governance Domain | Maturity Score (0-100) | Open High/Critical Findings | Status |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for d in domain_maturity:
        maturity_lines.append(
            f"| {d['domain']} | {d['maturity_score']:.1f} | {d['open_high_risk_findings']} | {d['status']} |"
        )

    cm = champ["confusion_matrix"]
    fp_dept_str = ", ".join(f"{k}: {v}" for k, v in err["fp_by_department"].items()) or "None"
    fn_type_str = ", ".join(f"{k}: {v}" for k, v in err["fn_by_violation_type"].items()) or "None"

    content = f"""# FinGuard-GRC — Technical Evaluation & Audit Benchmark Report

> Auto-generated from deterministic pipeline run (`seed=42`). All metrics are measured on a strict **3-way chronological split** (Days 0–{split['train_end_day']} Train / Days {split['train_end_day']+1}–{split['val_end_day']} Validation / Days {split['val_end_day']+1}–89 Holdout Test).

## 1. Dataset & 3-Way Chronological Split Summary
- **Total Audited Events**: {split['total_events']:,}
- **Training Split**: {split['train_events']:,} events (Days 0–{split['train_end_day']}, violation rate: {split['train_violation_rate']*100:.2f}%)
- **Validation Split** *(threshold calibration)*: {split['val_events']:,} events (Days {split['train_end_day']+1}–{split['val_end_day']}, violation rate: {split['val_violation_rate']*100:.2f}%)
- **Holdout Test Split** *(final evaluation)*: {split['test_events']:,} events (Days {split['val_end_day']+1}–89, violation rate: {split['test_violation_rate']*100:.2f}%)
- **Leakage Controls**: `ColumnTransformer` scalers and encoders are fit exclusively on Training data; F2-maximizing operating thresholds are tuned strictly on Validation probabilities; downstream compliance mapping and maturity scoring use zero ground-truth labels.

---

## 2. Five-Model Comparative Benchmark (Holdout Test Set)

{chr(10).join(table_lines)}

![Model Comparison](assets/model_comparison.svg)

![ROC and PR Curves](assets/pr_roc_curves.svg)

---

## 3. Champion Model Explainability & Dynamic Error Analysis
- **Selected Production Model**: `{champ['model_name']}` (Validation-calibrated threshold = `{champ['threshold']}`)
- **Test Confusion Matrix**: `TP={cm['tp']}`, `FP={cm['fp']}`, `FN={cm['fn']}`, `TN={cm['tn']}`
- **False Positive Burden**: {err['false_positives_count']} benign events flagged (`FPR={champ['fpr']:.4f}`). Distribution by department: {fp_dept_str}.
  - *Observed FP Drivers*: {'; '.join(err['fp_sample_causes'])}.
- **False Negative Misses**: {err['false_negatives_count']} stealth violations missed. Distribution by archetype: {fn_type_str}.
  - *Observed FN Drivers*: {'; '.join(err['fn_sample_causes'])}.

![Permutation Feature Importance](assets/feature_importance.svg)

---

## 4. Hybrid Regulatory Compliance RAG Ablation Study
Evaluated across **{rag_eval['num_queries']}** paraphrased financial IT audit queries against HKMA SPM TM-G-1, HKMA C-RAF, ISO/IEC 27001:2022, and NIST AI RMF 1.0:

{chr(10).join(ablation_lines)}

---

## 5. Five-Domain IT Governance Maturity Snapshot

{chr(10).join(maturity_lines)}

![Governance Risk Matrix](assets/governance_risk_matrix.svg)
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path
