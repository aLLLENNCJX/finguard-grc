"""Tests for hybrid BM25 + TF-IDF compliance RAG, ablation, and governance simulation."""

from __future__ import annotations

from finguard.compliance.governance import compute_domain_maturity, simulate_remediation
from finguard.compliance.rag_retriever import HybridComplianceRetriever, evaluate_retriever
from finguard.data.generator import generate_audit_events
from finguard.models.scoring import score_events


def test_hybrid_rag_retrieval_and_ablation() -> None:
    retriever = HybridComplianceRetriever()
    eval_metrics = evaluate_retriever(retriever)

    assert eval_metrics["num_queries"] == 24
    assert eval_metrics["recall_at_3"] >= 0.80
    assert eval_metrics["mrr"] >= 0.65
    assert len(eval_metrics["ablation"]) == 3

    hits = retriever.query("developer uploaded client KYC data to unapproved external LLM API", top_k=3)
    assert len(hits) == 3
    top_ids = [h["control_id"] for h in hits]
    assert any(cid in top_ids for cid in ["HKMA-TMG1-7.4", "NIST-AI-GOVERN-1.2", "NIST-AI-MAP-1.5"])

    # map_finding_controls must work strictly from observable text + reason codes
    mapped = retriever.map_finding_controls(
        "EMP-1001 performed config_change on Production IAM Console (ticket=0)",
        ["Production configuration change without ITSM ticket"],
        top_k=2,
    )
    assert len(mapped) == 2


def test_governance_maturity_without_labels_and_domain_focus_simulation() -> None:
    df = generate_audit_events(num_events=300, num_days=20, seed=42)
    probs = [0.88 if v == 1 else 0.08 for v in df["is_violation"]]
    scored = score_events(df, probs)

    # Drop violation_type to prove compute_domain_maturity has zero label leakage
    scored_no_vtype = scored.drop(columns=["violation_type"])
    maturity = compute_domain_maturity(scored_no_vtype)
    assert len(maturity) == 5
    assert all(0.0 <= d["maturity_score"] <= 100.0 for d in maturity)

    sim_all = simulate_remediation(scored, alert_threshold=0.45, budget_hours=160, domain_focus="All")
    sim_domain = simulate_remediation(scored, alert_threshold=0.45, budget_hours=160, domain_focus="Access & Identity")
    assert sim_domain["domain_focus"] == "Access & Identity"
    assert sim_domain["flagged_alerts"] <= sim_all["flagged_alerts"]