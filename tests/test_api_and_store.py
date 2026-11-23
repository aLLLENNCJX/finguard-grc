"""Tests for SQLite AuditStore and Flask REST API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from finguard.api.app import create_app
from finguard.api.store import AuditStore
from finguard.data.generator import generate_audit_events
from finguard.models.scoring import score_events


def _seed_store(tmp_dir: Path) -> AuditStore:
    df = generate_audit_events(num_events=80, num_days=10, seed=42)
    probs = [0.85 if v == 1 else 0.08 for v in df["is_violation"]]
    scored = score_events(df, probs)
    scored["split_role"] = ["test" if i >= 50 else "train" for i in range(len(scored))]
    scored["mapped_controls"] = [["HKMA-TMG1-4.1"] for _ in range(len(scored))]

    store = AuditStore(db_path=tmp_dir / "finguard_audit.db")
    store.save_findings(scored)
    (tmp_dir / "dashboard_snapshot.json").write_text(
        json.dumps({"summary": {"total_events": len(scored)}, "rag_evaluation": {"recall_at_3": 0.95}}),
        encoding="utf-8",
    )
    (tmp_dir / "benchmark_results.json").write_text(
        json.dumps({"benchmarks": [], "error_analysis": {}}),
        encoding="utf-8",
    )
    return store


def test_flask_api_endpoints_brief_and_validation(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    app = create_app(db_path=tmp_path / "finguard_audit.db", generated_dir=tmp_path)
    client = app.test_client()

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/summary").status_code == 200

    # Invalid integer query param returns clean 400 JSON error (not 500)
    bad_res = client.get("/api/events?limit=abc")
    assert bad_res.status_code == 400

    events_res = client.get("/api/events?limit=10")
    assert events_res.status_code == 200
    events_data = events_res.get_json()
    assert events_data["count"] == 10
    first_id = events_data["events"][0]["event_id"]

    # Audit brief endpoint
    brief_res = client.post(f"/api/events/{first_id}/brief")
    assert brief_res.status_code == 200
    assert "brief" in brief_res.get_json()

    # Record human-in-the-loop review
    rev_res = client.post(
        f"/api/events/{first_id}/review",
        json={"status": "Escalated to CISO", "reviewer": "Lead Auditor", "comment": "Verified SoD issue"},
    )
    assert rev_res.status_code == 200
    assert rev_res.get_json()["finding"]["review_status"] == "Escalated to CISO"
    assert len(rev_res.get_json()["finding"]["review_history"]) == 1
    assert rev_res.get_json()["review_stats"]["reviewed_findings"] == 1

    # Compliance RAG query
    rag_res = client.post("/api/compliance/query", json={"query": "unapproved production change", "top_k": 2})
    assert rag_res.status_code == 200
    assert len(rag_res.get_json()["results"]) == 2

    # Simulation endpoint with domain_focus
    sim_res = client.post(
        "/api/simulate/remediation",
        json={"alert_threshold": 0.45, "budget_hours": 160, "domain_focus": "Access & Identity"},
    )
    assert sim_res.status_code == 200
    assert sim_res.get_json()["domain_focus"] == "Access & Identity"