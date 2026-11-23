"""Command-line interface for FinGuard-GRC pipeline and web dashboard."""

from __future__ import annotations

import argparse
import json
from typing import Any
import joblib

from finguard.api.app import create_app
from finguard.api.store import AuditStore
from finguard.compliance.governance import (
    compute_domain_maturity,
    compute_risk_heatmap,
    simulate_remediation,
)
from finguard.compliance.rag_retriever import (
    HybridComplianceRetriever,
    evaluate_retriever,
)
from finguard.config import (
    DEFAULT_NUM_DAYS,
    DEFAULT_NUM_EVENTS,
    DEFAULT_RANDOM_SEED,
    GENERATED_DIR,
    KNOWLEDGE_BASE_PATH,
    ensure_directories,
)
from finguard.data.generator import generate_audit_events
from finguard.features.engineering import get_feature_columns, prepare_feature_frame
from finguard.models.benchmark import run_benchmark_and_train
from finguard.models.scoring import score_events
from finguard.reporting.charts import generate_all_charts
from finguard.reporting.report_builder import build_evaluation_report
from finguard.reporting.static_export import export_static_dashboard


def run_full_pipeline(verify_only: bool = False) -> dict[str, Any]:
    """Execute end-to-end deterministic data, ML, RAG, DB, chart & static export pipeline."""
    ensure_directories()

    # 1. Generate synthetic enterprise IT audit telemetry
    raw_df = generate_audit_events(
        num_events=DEFAULT_NUM_EVENTS,
        num_days=DEFAULT_NUM_DAYS,
        seed=DEFAULT_RANDOM_SEED,
    )
    raw_df.to_csv(GENERATED_DIR / "audit_events.csv", index=False, encoding="utf-8")

    # 2. Train & benchmark 5 models on 3-way chronological split (Train / Val / Holdout Test)
    benchmark_results = run_benchmark_and_train(raw_df, seed=DEFAULT_RANDOM_SEED, output_dir=GENERATED_DIR)
    split_info = benchmark_results["split_info"]
    train_end = int(split_info["train_end_day"])
    val_end = int(split_info["val_end_day"])

    # 3. Score all events using champion model & composite IT governance risk score
    model_bundle = joblib.load(GENERATED_DIR / "champion_model.joblib")
    champion_pipe = model_bundle["pipeline"]
    feat_df = prepare_feature_frame(raw_df)
    ml_probs = champion_pipe.predict_proba(feat_df[get_feature_columns()])[:, 1]
    scored_df = score_events(raw_df, ml_probs)
    scored_df["split_role"] = [
        "train" if d <= train_end else ("val" if d <= val_end else "test")
        for d in scored_df["day_index"]
    ]

    # 4. Map each finding strictly from observable summary & reason codes (zero label leakage)
    retriever = HybridComplianceRetriever(KNOWLEDGE_BASE_PATH)
    rag_eval = evaluate_retriever(retriever)

    scored_df["mapped_controls"] = [
        retriever.map_finding_controls(row["event_summary"], row["reason_codes"], top_k=2)
        for _, row in scored_df.iterrows()
    ]

    # 5. Compute Governance Maturity, Heatmap & Out-of-Sample Holdout Simulation
    domain_maturity = compute_domain_maturity(scored_df)
    heatmap = compute_risk_heatmap(scored_df)
    holdout_df = scored_df[scored_df["split_role"] == "test"].copy()
    default_sim = simulate_remediation(
        holdout_df,
        alert_threshold=benchmark_results["champion_threshold"],
        budget_hours=160,
        quarterly_scale_factor=90.0 / 23.0,
    )

    # 6. Persist to SQLite AuditStore
    store = AuditStore(GENERATED_DIR / "finguard_audit.db")
    store.save_findings(scored_df)
    store.set_metadata("benchmark_results", benchmark_results)
    store.set_metadata("rag_evaluation", rag_eval)

    tier_counts = scored_df["risk_tier"].value_counts().to_dict()
    champ_row = benchmark_results["benchmarks"][-1]
    overall_maturity = round(
        sum(d["maturity_score"] for d in domain_maturity) / max(1, len(domain_maturity)),
        1,
    )

    summary_payload = {
        "total_events": int(len(scored_df)),
        "critical_findings": int(tier_counts.get("Critical", 0)),
        "high_findings": int(tier_counts.get("High", 0)),
        "medium_findings": int(tier_counts.get("Medium", 0)),
        "low_findings": int(tier_counts.get("Low", 0)),
        "overall_maturity": overall_maturity,
        "alert_compression_pct": round(champ_row["alert_compression_ratio"] * 100.0, 1),
        "analyst_hours_saved_quarter": champ_row["analyst_hours_saved_quarter"],
        "avoided_exposure_hkd": default_sim["avoided_exposure_hkd"],
        "domain_maturity": domain_maturity,
        "heatmap": heatmap,
        "tier_distribution": {str(k): int(v) for k, v in tier_counts.items()},
        "review_stats": store.get_review_stats(),
    }

    # Compact holdout test corpus [prob, is_violation, domain_idx] so static GitHub Pages ROI simulator matches Flask API 100%
    domain_idx_map = {
        "Access & Identity": 0,
        "Change & Operations": 1,
        "Data Protection": 2,
        "AI & Third-Party Governance": 3,
        "Monitoring & Audit": 4,
    }
    simulation_corpus = [
        [
            round(float(r["ml_probability"]), 4),
            int(r["is_violation"]),
            domain_idx_map.get(str(r["primary_domain"]), 4),
        ]
        for _, r in holdout_df.iterrows()
    ]

    top_events = store.query_events(limit=180)
    controls_raw = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    snapshot = {
        "summary": summary_payload,
        "events": top_events,
        "evaluation": benchmark_results,
        "rag_evaluation": rag_eval,
        "controls": controls_raw,
        "default_simulation": default_sim,
        "simulation_corpus": simulation_corpus,
    }
    (GENERATED_DIR / "dashboard_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )

    if not verify_only:
        generate_all_charts(benchmark_results, scored_df, domain_maturity)
        build_evaluation_report(benchmark_results, rag_eval, domain_maturity)
        static_snapshot = {
            **snapshot,
            "events": top_events[:60],
        }
        export_static_dashboard(static_snapshot)

    return {
        "split": split_info,
        "champion": champ_row,
        "rag_evaluation": rag_eval,
        "overall_maturity": overall_maturity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FinGuard-GRC CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run-all", help="Run full data, ML, RAG, charts, and static export pipeline")
    run_p.add_argument("--verify-only", action="store_true", help="Skip chart/HTML export for fast CI verification")

    serve_p = sub.add_parser("serve", help="Start interactive Flask REST API & Web Dashboard")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()
    if args.command == "run-all":
        summary = run_full_pipeline(verify_only=args.verify_only)
        print(json.dumps(summary, indent=2))
    elif args.command == "serve":
        if not (GENERATED_DIR / "dashboard_snapshot.json").exists():
            run_full_pipeline(verify_only=False)
        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
