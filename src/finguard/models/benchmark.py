"""5-Model Comparative Benchmark, 3-Way Chronological Evaluation & Dynamic Error Analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from finguard.config import DEFAULT_RANDOM_SEED, GENERATED_DIR, TRAIN_SPLIT_RATIO
from finguard.features.engineering import (
    build_preprocessor,
    get_feature_columns,
    prepare_feature_frame,
)


def heuristic_siem_scores(df: pd.DataFrame) -> np.ndarray:
    """Deterministic enterprise SIEM rule baseline (prone to edge-case FPs/FNs)."""
    scores = np.zeros(len(df), dtype=float)
    scores += (df["is_off_hours"].values == 1) * 0.28
    scores += (df["records_exported"].values >= 3500) * 0.28
    scores += (df["has_change_ticket"].values == 0) * 0.18
    scores += (df["sod_rule_score"].values >= 0.58) * 0.30
    scores += (df["external_ai_endpoint_flag"].values == 1) * 0.35
    scores += (df["dormancy_days"].values >= 35) * 0.30
    scores += (df["failed_auth_24h"].values >= 2) * 0.20
    return np.clip(scores, 0.0, 1.0)


def _select_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select operating threshold maximizing F2-score (audit recall-prioritized)."""
    best_thresh = 0.50
    best_f2 = -1.0
    for thresh in np.linspace(0.20, 0.80, 61):
        preds = (y_prob >= thresh).astype(int)
        score = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_thresh = float(round(thresh, 3))
    return best_thresh


def _compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, fp + tn))
    total_events = len(y_true)
    flagged = int(y_pred.sum())
    compression = float(1.0 - (flagged / max(1, total_events)))
    baseline_alerts = max(flagged, int((y_true.sum() * 2.6)))
    hours_saved_quarter = round(max(0.0, (baseline_alerts - fp) * 0.25 * 3.0), 1)

    return {
        "model_name": model_name,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)), 4),
        "fpr": round(fpr, 4),
        "alert_compression_ratio": round(compression, 4),
        "analyst_hours_saved_quarter": hours_saved_quarter,
        "threshold": round(float(threshold), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _downsample_curve(x_arr: np.ndarray, y_arr: np.ndarray, max_points: int = 45) -> list[dict[str, float]]:
    if len(x_arr) <= max_points:
        indices = np.arange(len(x_arr))
    else:
        indices = np.linspace(0, len(x_arr) - 1, max_points, dtype=int)
    return [{"x": round(float(x_arr[i]), 4), "y": round(float(y_arr[i]), 4)} for i in indices]


def _build_dynamic_error_causes(fp_df: pd.DataFrame, fn_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Summarize actual observed False Positive & False Negative patterns from holdout errors."""
    fp_causes: list[str] = []
    if len(fp_df) > 0:
        ticketed_off_hours = int(((fp_df["is_off_hours"] == 1) & (fp_df["has_change_ticket"] == 1)).sum())
        bulk_compliance = int(((fp_df["records_exported"] >= 3500) & (fp_df["department"] == "Risk & Compliance")).sum())
        elevated_sod_benign = int((fp_df["sod_rule_score"] >= 0.45).sum())
        if ticketed_off_hours > 0:
            fp_causes.append(f"{ticketed_off_hours}/{len(fp_df)} FPs had off-hours activity despite valid ITSM change tickets")
        if bulk_compliance > 0:
            fp_causes.append(f"{bulk_compliance}/{len(fp_df)} FPs were Risk & Compliance bulk exports (z-score={fp_df['peer_group_volume_zscore'].median():.2f})")
        if elevated_sod_benign > 0:
            fp_causes.append(f"{elevated_sod_benign}/{len(fp_df)} FPs exhibited temporary cross-dept SoD score >= 0.45")
    if not fp_causes:
        fp_causes.append("No holdout false positives observed at the selected threshold")

    fn_causes: list[str] = []
    if len(fn_df) > 0:
        top_types = fn_df["violation_type"].value_counts().head(2).to_dict()
        type_str = ", ".join(f"{k} ({v})" for k, v in top_types.items())
        fn_causes.append(f"Missed archetypes: {type_str}; median ML prob = {fn_df['prob'].median():.3f}")
        fn_causes.append(
            f"Stealth profile: median exported={int(fn_df['records_exported'].median())} records, "
            f"median SoD={fn_df['sod_rule_score'].median():.2f}, ticket_present={int((fn_df['has_change_ticket'] == 1).sum())}/{len(fn_df)}"
        )
    else:
        fn_causes.append("No holdout false negatives observed at the selected threshold")

    return fp_causes, fn_causes


def run_benchmark_and_train(
    raw_df: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
    output_dir: Path = GENERATED_DIR,
) -> dict[str, Any]:
    """Run strict 3-way chronological split (Train / Val / Holdout Test), benchmark 5 models, and persist champion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_df = prepare_feature_frame(raw_df)
    feature_cols = get_feature_columns()

    # 3-way chronological split: Train (0..60%), Validation (60..75% for threshold tuning), Test (75..100% holdout)
    max_day = int(feat_df["day_index"].max())
    train_end_day = int(max_day * 0.60)
    val_end_day = int(max_day * TRAIN_SPLIT_RATIO)

    train_df = feat_df.loc[feat_df["day_index"] <= train_end_day].reset_index(drop=True)
    val_df = feat_df.loc[(feat_df["day_index"] > train_end_day) & (feat_df["day_index"] <= val_end_day)].reset_index(drop=True)
    test_df = feat_df.loc[feat_df["day_index"] > val_end_day].reset_index(drop=True)

    X_train, y_train = train_df[feature_cols], train_df["is_violation"].values.astype(int)
    X_val, y_val = val_df[feature_cols], val_df["is_violation"].values.astype(int)
    X_test, y_test = test_df[feature_cols], test_df["is_violation"].values.astype(int)

    # 1. Rule-Based Heuristic Baseline
    rule_probs_test = heuristic_siem_scores(test_df)
    rule_metrics = _compute_metrics("Rule-Based SIEM Baseline", y_test, rule_probs_test, threshold=0.45)

    # 2. Unsupervised Isolation Forest (fit on X_train only, threshold tuned on Validation)
    iso_pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", IsolationForest(n_estimators=180, contamination=0.10, random_state=seed)),
        ]
    )
    iso_pipe.fit(X_train)
    raw_iso_val = -iso_pipe.decision_function(X_val)
    iso_min, iso_max = float(raw_iso_val.min()), float(raw_iso_val.max())
    iso_probs_val = np.clip((raw_iso_val - iso_min) / max(1e-9, iso_max - iso_min), 0.0, 1.0)
    iso_thresh = _select_f2_threshold(y_val, iso_probs_val)

    raw_iso_test = -iso_pipe.decision_function(X_test)
    iso_probs_test = np.clip((raw_iso_test - iso_min) / max(1e-9, iso_max - iso_min), 0.0, 1.0)
    iso_metrics = _compute_metrics("Isolation Forest (Unsupervised)", y_test, iso_probs_test, threshold=iso_thresh)

    # 3. Logistic Regression (L2, Balanced)
    lr_pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    lr_thresh = _select_f2_threshold(y_val, lr_pipe.predict_proba(X_val)[:, 1])
    lr_probs_test = lr_pipe.predict_proba(X_test)[:, 1]
    lr_metrics = _compute_metrics("Logistic Regression (Balanced)", y_test, lr_probs_test, threshold=lr_thresh)

    # 4. Random Forest Classifier
    rf_pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=seed,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    rf_thresh = _select_f2_threshold(y_val, rf_pipe.predict_proba(X_val)[:, 1])
    rf_probs_test = rf_pipe.predict_proba(X_test)[:, 1]
    rf_metrics = _compute_metrics("Random Forest Classifier", y_test, rf_probs_test, threshold=rf_thresh)

    # 5. HistGradientBoosting Classifier (Champion Production Model)
    hgb_pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.06,
                    max_depth=6,
                    min_samples_leaf=18,
                    l2_regularization=1.2,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    hgb_pipe.fit(X_train, y_train)
    hgb_thresh = _select_f2_threshold(y_val, hgb_pipe.predict_proba(X_val)[:, 1])
    hgb_probs_test = hgb_pipe.predict_proba(X_test)[:, 1]
    hgb_metrics = _compute_metrics(
        "HistGradientBoosting (Champion)", y_test, hgb_probs_test, threshold=hgb_thresh
    )

    benchmark_rows = [rule_metrics, iso_metrics, lr_metrics, rf_metrics, hgb_metrics]

    model_probs_map = {
        "Rule-Based SIEM Baseline": rule_probs_test,
        "Isolation Forest (Unsupervised)": iso_probs_test,
        "Logistic Regression (Balanced)": lr_probs_test,
        "Random Forest Classifier": rf_probs_test,
        "HistGradientBoosting (Champion)": hgb_probs_test,
    }
    curves: dict[str, dict[str, list[dict[str, float]]]] = {}
    for name, probs in model_probs_map.items():
        fpr_arr, tpr_arr, _ = roc_curve(y_test, probs)
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs)
        curves[name] = {
            "roc": _downsample_curve(fpr_arr, tpr_arr),
            "pr": _downsample_curve(rec_arr[::-1], prec_arr[::-1]),
        }

    perm = permutation_importance(
        hgb_pipe, X_test, y_test, n_repeats=8, random_state=seed, scoring="average_precision"
    )
    feat_imp = sorted(
        [
            {
                "feature": col,
                "importance_mean": round(float(perm.importances_mean[idx]), 4),
                "importance_std": round(float(perm.importances_std[idx]), 4),
            }
            for idx, col in enumerate(feature_cols)
        ],
        key=lambda item: item["importance_mean"],
        reverse=True,
    )

    # Dynamic Error Analysis on Holdout Test Set
    y_pred_champ = (hgb_probs_test >= hgb_thresh).astype(int)
    eval_test = test_df.copy()
    eval_test["pred"] = y_pred_champ
    eval_test["prob"] = hgb_probs_test

    fp_df = eval_test[(eval_test["is_violation"] == 0) & (eval_test["pred"] == 1)]
    fn_df = eval_test[(eval_test["is_violation"] == 1) & (eval_test["pred"] == 0)]
    fp_causes, fn_causes = _build_dynamic_error_causes(fp_df, fn_df)

    error_analysis = {
        "test_size": int(len(test_df)),
        "test_violations": int(y_test.sum()),
        "false_positives_count": int(len(fp_df)),
        "false_negatives_count": int(len(fn_df)),
        "fp_by_department": {str(k): int(v) for k, v in fp_df["department"].value_counts().to_dict().items()},
        "fn_by_violation_type": {str(k): int(v) for k, v in fn_df["violation_type"].value_counts().to_dict().items()},
        "fp_sample_causes": fp_causes,
        "fn_sample_causes": fn_causes,
    }

    model_path = output_dir / "champion_model.joblib"
    joblib.dump({"pipeline": hgb_pipe, "threshold": hgb_thresh, "feature_cols": feature_cols}, model_path)

    results = {
        "split_info": {
            "total_events": int(len(feat_df)),
            "train_events": int(len(train_df)),
            "val_events": int(len(val_df)),
            "test_events": int(len(test_df)),
            "train_end_day": train_end_day,
            "val_end_day": val_end_day,
            "cutoff_day": val_end_day,
            "train_violation_rate": round(float(y_train.mean()), 4),
            "val_violation_rate": round(float(y_val.mean()), 4),
            "test_violation_rate": round(float(y_test.mean()), 4),
        },
        "benchmarks": benchmark_rows,
        "curves": curves,
        "feature_importance": feat_imp,
        "error_analysis": error_analysis,
        "champion_threshold": hgb_thresh,
    }

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
