"""SQLite persistence & immutable human-in-the-loop audit trail store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from finguard.config import GENERATED_DIR
from finguard.schemas import ReviewDecision


class AuditStore:
    """Manages scored audit findings, governance metadata, and immutable triage reviews."""

    def __init__(self, db_path: Path = GENERATED_DIR / "finguard_audit.db") -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # Handle clean migration if an older schema lacks primary_domain / split_role
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(findings)").fetchall()
            }
            if cols and ("primary_domain" not in cols or "split_role" not in cols):
                conn.execute("DROP TABLE IF EXISTS findings")

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS findings (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    day_index INTEGER NOT NULL,
                    split_role TEXT NOT NULL DEFAULT 'test',
                    employee_id TEXT NOT NULL,
                    department TEXT NOT NULL,
                    system_name TEXT NOT NULL,
                    asset_criticality INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    is_violation INTEGER NOT NULL,
                    ml_probability REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_tier TEXT NOT NULL,
                    primary_domain TEXT NOT NULL DEFAULT 'Monitoring & Audit',
                    reason_codes_json TEXT NOT NULL,
                    mapped_controls_json TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'Pending Review',
                    event_summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES findings(event_id)
                );

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )

    def save_findings(self, df: pd.DataFrame) -> None:
        """Persist scored audit findings (preserves any existing human review status)."""
        with self._connect() as conn:
            existing_reviews = {
                row["event_id"]: row["review_status"]
                for row in conn.execute("SELECT event_id, review_status FROM findings").fetchall()
                if row["review_status"] != "Pending Review"
            }
            conn.execute("DELETE FROM findings")
            rows = []
            for _, r in df.iterrows():
                eid = str(r["event_id"])
                status = existing_reviews.get(eid, "Pending Review")
                rows.append(
                    (
                        eid,
                        str(r["timestamp"]),
                        int(r["day_index"]),
                        str(r.get("split_role", "test")),
                        str(r["employee_id"]),
                        str(r["department"]),
                        str(r["system_name"]),
                        int(r["asset_criticality"]),
                        str(r["action_type"]),
                        str(r["violation_type"]),
                        int(r["is_violation"]),
                        float(r["ml_probability"]),
                        float(r["risk_score"]),
                        str(r["risk_tier"]),
                        str(r.get("primary_domain", "Monitoring & Audit")),
                        json.dumps(list(r["reason_codes"])),
                        json.dumps(list(r["mapped_controls"])),
                        status,
                        str(r["event_summary"]),
                    )
                )
            conn.executemany(
                """
                INSERT INTO findings (
                    event_id, timestamp, day_index, split_role, employee_id, department,
                    system_name, asset_criticality, action_type, violation_type,
                    is_violation, ml_probability, risk_score, risk_tier, primary_domain,
                    reason_codes_json, mapped_controls_json, review_status, event_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def set_metadata(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM metadata WHERE key = ?", (key,)).fetchone()
            return json.loads(row["value_json"]) if row else default

    def _hydrate_row(self, row: sqlite3.Row, history: list[dict[str, Any]]) -> dict[str, Any]:
        item = dict(row)
        item["reason_codes"] = json.loads(item.pop("reason_codes_json"))
        item["mapped_controls"] = json.loads(item.pop("mapped_controls_json"))
        item["review_history"] = history
        return item

    def get_event_by_id(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM findings WHERE event_id = ?", (event_id,)).fetchone()
            if not row:
                return None
            history = conn.execute(
                "SELECT event_id, status, reviewer, comment, created_at FROM audit_reviews WHERE event_id = ? ORDER BY review_id DESC",
                (event_id,),
            ).fetchall()
        return self._hydrate_row(row, [dict(h) for h in history])

    def get_review_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT review_status, COUNT(*) AS cnt FROM findings GROUP BY review_status"
            ).fetchall()
            total_reviews = conn.execute("SELECT COUNT(*) AS cnt FROM audit_reviews").fetchone()["cnt"]
        by_status = {str(r["review_status"]): int(r["cnt"]) for r in rows}
        reviewed_findings = sum(cnt for st, cnt in by_status.items() if st != "Pending Review")
        return {
            "reviewed_findings": reviewed_findings,
            "total_review_actions": int(total_reviews),
            "by_status": by_status,
        }

    def query_events(
        self,
        tier: str | None = None,
        department: str | None = None,
        status: str | None = None,
        violation_type: str | None = None,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if tier and tier != "All":
            clauses.append("risk_tier = ?")
            params.append(tier)
        if department and department != "All":
            clauses.append("department = ?")
            params.append(department)
        if status and status != "All":
            clauses.append("review_status = ?")
            params.append(status)
        if violation_type and violation_type != "All":
            clauses.append("violation_type = ?")
            params.append(violation_type)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM findings {where_sql} ORDER BY risk_score DESC, timestamp DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            reviews_rows = conn.execute(
                "SELECT event_id, status, reviewer, comment, created_at FROM audit_reviews ORDER BY review_id DESC"
            ).fetchall()

        reviews_by_event: dict[str, list[dict[str, Any]]] = {}
        for rev in reviews_rows:
            reviews_by_event.setdefault(rev["event_id"], []).append(dict(rev))

        return [
            self._hydrate_row(row, reviews_by_event.get(row["event_id"], []))
            for row in rows
        ]

    def record_review(self, event_id: str, decision: ReviewDecision) -> dict[str, Any] | None:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT event_id FROM findings WHERE event_id = ?", (event_id,)
            ).fetchone()
            if not exists:
                return None
            conn.execute(
                "INSERT INTO audit_reviews (event_id, status, reviewer, comment, created_at) VALUES (?, ?, ?, ?, ?)",
                (event_id, decision.status, decision.reviewer, decision.comment, now_utc),
            )
            conn.execute(
                "UPDATE findings SET review_status = ? WHERE event_id = ?",
                (decision.status, event_id),
            )
        return self.get_event_by_id(event_id)

    def load_all_findings_df(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM findings", conn)
