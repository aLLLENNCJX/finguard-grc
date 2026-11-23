"""Publication-grade Matplotlib/Seaborn charts for README & evaluation report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from finguard.config import DOCS_ASSETS_DIR


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )


def generate_all_charts(
    benchmark_results: dict[str, Any],
    scored_df: pd.DataFrame,
    domain_maturity: list[dict[str, Any]],
    assets_dir: Path = DOCS_ASSETS_DIR,
) -> list[Path]:
    """Generate 4 high-DPI visual charts in docs/assets/."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    saved: list[Path] = []

    # 1. Model Comparison Grouped Bar Chart
    rows = benchmark_results["benchmarks"]
    short_names = [
        r["model_name"]
        .replace(" (Unsupervised)", "\n(Unsup.)")
        .replace(" (Balanced)", "\n(Balanced)")
        .replace(" (Champion)", "\n(Champion)")
        for r in rows
    ]
    metrics_keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1-Score"), ("recall", "Recall")]
    x = np.arange(len(short_names))
    width = 0.19
    palette = ["#1d4ed8", "#0f766e", "#d97706", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, (key, label) in enumerate(metrics_keys):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=palette[i], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("Score (Chronological Test Split)")
    ax.set_title("FinGuard-GRC: 5-Model IT Audit Anomaly Detection Benchmark")
    ax.legend(loc="upper left", ncol=4, frameon=True, facecolor="white")
    plt.tight_layout()
    p1 = assets_dir / "model_comparison.png"
    fig.savefig(p1, dpi=180)
    fig.savefig(assets_dir / "model_comparison.svg", format="svg")
    plt.close(fig)
    saved.append(p1)

    # 2. Side-by-side ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    curve_colors = ["#64748b", "#0891b2", "#2563eb", "#7c3aed", "#dc2626"]
    for idx, row in enumerate(rows):
        name = row["model_name"]
        c_data = benchmark_results["curves"][name]
        roc_pts = c_data["roc"]
        pr_pts = c_data["pr"]
        lw = 2.4 if "Champion" in name else 1.5
        ax1.plot(
            [p["x"] for p in roc_pts],
            [p["y"] for p in roc_pts],
            label=f"{name.split(' (')[0]} ({row['roc_auc']:.3f})",
            color=curve_colors[idx % len(curve_colors)],
            linewidth=lw,
        )
        ax2.plot(
            [p["x"] for p in pr_pts],
            [p["y"] for p in pr_pts],
            label=f"{name.split(' (')[0]} ({row['pr_auc']:.3f})",
            color=curve_colors[idx % len(curve_colors)],
            linewidth=lw,
        )

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1)
    ax1.set_title("ROC Curves (Test Set)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")

    ax2.set_title("Precision-Recall Curves (Imbalanced Audit Alerts)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white")

    plt.tight_layout()
    p2 = assets_dir / "pr_roc_curves.png"
    fig.savefig(p2, dpi=180)
    fig.savefig(assets_dir / "pr_roc_curves.svg", format="svg")
    plt.close(fig)
    saved.append(p2)

    # 3. Top 12 Permutation Feature Importances
    top_feats = benchmark_results["feature_importance"][:12][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    feat_names = [f["feature"] for f in top_feats]
    imp_vals = [max(0.0, f["importance_mean"]) for f in top_feats]
    imp_errs = [f["importance_std"] for f in top_feats]
    ax.barh(feat_names, imp_vals, xerr=imp_errs, color="#0f766e", alpha=0.9, capsize=3)
    ax.set_xlabel("Mean PR-AUC Drop on Permutation (Test Set)")
    ax.set_title("Top 12 Risk Drivers — Champion Model Permutation Importance")
    plt.tight_layout()
    p3 = assets_dir / "feature_importance.png"
    fig.savefig(p3, dpi=180)
    fig.savefig(assets_dir / "feature_importance.svg", format="svg")
    plt.close(fig)
    saved.append(p3)

    # 4. Governance Risk Matrix & 5-Domain Maturity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0), gridspec_kw={"width_ratios": [1.45, 1.0]})
    heat = scored_df.pivot_table(
        index="department", columns="system_name", values="risk_score", aggfunc="mean"
    ).fillna(0.0)
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax1,
        cbar_kws={"label": "Mean Risk Score (0-100)"},
    )
    ax1.set_title("IT Governance Risk Heatmap (Department × System)")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", rotation=32, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8.5)

    domains = [d["domain"] for d in domain_maturity][::-1]
    scores = [d["maturity_score"] for d in domain_maturity][::-1]
    bar_colors = ["#059669" if s >= 80 else ("#d97706" if s >= 65 else "#dc2626") for s in scores]
    ax2.barh(domains, scores, color=bar_colors, height=0.55)
    ax2.set_xlim(0, 105)
    ax2.axvline(80, color="#059669", linestyle="--", alpha=0.6, linewidth=1)
    for i, s in enumerate(scores):
        ax2.text(s + 1.5, i, f"{s:.1f}", va="center", fontsize=9, fontweight="bold")
    ax2.set_title("5-Domain Control Maturity (0-100)")
    ax2.set_xlabel("Maturity Score")

    plt.tight_layout()
    p4 = assets_dir / "governance_risk_matrix.png"
    fig.savefig(p4, dpi=180)
    fig.savefig(assets_dir / "governance_risk_matrix.svg", format="svg")
    plt.close(fig)
    saved.append(p4)

    return saved
