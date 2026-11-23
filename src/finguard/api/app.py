"""Flask REST API & Interactive Dashboard Server for FinGuard-GRC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from finguard.api.store import AuditStore
from finguard.compliance.governance import simulate_remediation
from finguard.compliance.llm_advisor import generate_audit_brief
from finguard.compliance.rag_retriever import HybridComplianceRetriever
from finguard.config import DASHBOARD_TEMPLATE_PATH, GENERATED_DIR, KNOWLEDGE_BASE_PATH
from finguard.schemas import RemediationScenario, ReviewDecision


def _parse_int_param(raw: Any, default: int, min_val: int, max_val: int) -> int:
    if raw is None or raw == "":
        return default
    val = int(raw)
    return min(max_val, max(min_val, val))


def create_app(
    db_path: Path | None = None,
    kb_path: Path | None = None,
    generated_dir: Path | None = None,
) -> Flask:
    """Create configured Flask application."""
    app = Flask(__name__)
    gen_dir = generated_dir or GENERATED_DIR
    store = AuditStore(db_path=db_path or (gen_dir / "finguard_audit.db"))
    retriever = HybridComplianceRetriever(kb_path=kb_path or KNOWLEDGE_BASE_PATH)

    def _load_snapshot() -> dict[str, Any]:
        snap_path = gen_dir / "dashboard_snapshot.json"
        if snap_path.exists():
            return json.loads(snap_path.read_text(encoding="utf-8"))
        return {}

    @app.get("/")
    def index() -> Response:
        html = DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
        html = html.replace(
            "/*__FINGUARD_BOOTSTRAP__*/",
            "window.__FINGUARD_API_MODE__ = true;",
        )
        return Response(html, mimetype="text/html; charset=utf-8")

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok", "service": "finguard-grc", "version": "1.0.0"})

    @app.get("/api/summary")
    def summary() -> Response:
        snap = _load_snapshot()
        summary_data = dict(snap.get("summary", {}))
        summary_data["review_stats"] = store.get_review_stats()
        return jsonify(summary_data)

    @app.get("/api/events")
    def events() -> tuple[Response, int] | Response:
        tier = request.args.get("tier")
        department = request.args.get("department")
        status = request.args.get("status")
        violation_type = request.args.get("violation_type")
        try:
            limit = _parse_int_param(request.args.get("limit"), default=120, min_val=1, max_val=500)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'limit' query parameter; expected integer."}), 400

        rows = store.query_events(
            tier=tier,
            department=department,
            status=status,
            violation_type=violation_type,
            limit=limit,
        )
        return jsonify({"count": len(rows), "events": rows})

    @app.post("/api/events/<event_id>/review")
    def review_event(event_id: str) -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        try:
            decision = ReviewDecision(**payload)
        except ValidationError as exc:
            return jsonify({"error": "Invalid review payload", "details": exc.errors()}), 400

        updated = store.record_review(event_id, decision)
        if updated is None:
            return jsonify({"error": f"Finding {event_id} not found"}), 404
        return jsonify({"status": "updated", "finding": updated, "review_stats": store.get_review_stats()})

    @app.post("/api/events/<event_id>/brief")
    def event_brief(event_id: str) -> tuple[Response, int] | Response:
        finding = store.get_event_by_id(event_id)
        if finding is None:
            return jsonify({"error": f"Finding {event_id} not found"}), 404
        query_text = f"{finding['event_summary']} {' '.join(finding.get('reason_codes', []))}"
        matched_controls = retriever.query(query_text, top_k=2, mode="hybrid")
        brief_res = generate_audit_brief(finding, matched_controls)
        return jsonify(
            {
                "event_id": event_id,
                "mode": brief_res["mode"],
                "brief": brief_res["brief"],
                "matched_controls": matched_controls,
            }
        )

    @app.post("/api/compliance/query")
    def compliance_query() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        query_text = str(payload.get("query", "")).strip()
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        try:
            top_k = _parse_int_param(payload.get("top_k"), default=3, min_val=1, max_val=10)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'top_k' parameter; expected integer."}), 400
        mode = str(payload.get("mode", "hybrid")).lower()
        if mode not in {"hybrid", "bm25", "tfidf"}:
            mode = "hybrid"
        hits = retriever.query(query_text, top_k=top_k, mode=mode)
        return jsonify({"query": query_text, "mode": mode, "results": hits})

    @app.get("/api/models/evaluation")
    def model_evaluation() -> Response:
        bench_path = gen_dir / "benchmark_results.json"
        data = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}
        snap = _load_snapshot()
        data["rag_evaluation"] = snap.get("rag_evaluation", {})
        return jsonify(data)

    @app.post("/api/simulate/remediation")
    def simulate() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True) or {}
        try:
            scenario = RemediationScenario(**payload)
        except ValidationError as exc:
            return jsonify({"error": "Invalid simulation parameters", "details": exc.errors()}), 400

        df = store.load_all_findings_df()
        if df.empty:
            return jsonify({"error": "No scored findings loaded in store"}), 404

        test_df = df[df["split_role"] == "test"] if "split_role" in df.columns else df
        if test_df.empty:
            test_df = df
        scale = 90.0 / 23.0 if len(test_df) < len(df) else 1.0
        result = simulate_remediation(
            test_df,
            alert_threshold=scenario.alert_threshold,
            budget_hours=scenario.budget_hours,
            domain_focus=scenario.domain_focus,
            quarterly_scale_factor=scale,
        )
        return jsonify(result)

    return app
