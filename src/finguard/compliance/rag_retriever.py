"""Deterministic Hybrid BM25 + TF-IDF Cosine Regulatory Compliance Retriever."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finguard.config import KNOWLEDGE_BASE_PATH
from finguard.schemas import ControlClause


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class HybridComplianceRetriever:
    """Offline hybrid BM25 + TF-IDF retriever over HKMA, ISO 27001, and NIST AI RMF."""

    def __init__(self, kb_path: Path = KNOWLEDGE_BASE_PATH) -> None:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
        self.controls: list[ControlClause] = [ControlClause(**item) for item in raw]
        self.corpus_texts: list[str] = [
            f"{c.control_id} {c.framework} {c.domain} {c.title} {c.description} "
            f"{' '.join(c.keywords)} {' '.join(c.remediation_steps)}"
            for c in self.controls
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)

        self.doc_tokens = [_tokenize(t) for t in self.corpus_texts]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.doc_freqs: Counter[str] = Counter()
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.doc_freqs[tok] += 1
        self.k1 = 1.5
        self.b = 0.75

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        n_docs = len(self.doc_tokens)
        scores = np.zeros(n_docs, dtype=float)
        for idx, toks in enumerate(self.doc_tokens):
            tf_map = Counter(toks)
            dl = self.doc_lens[idx]
            doc_score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                df = self.doc_freqs.get(qt, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map[qt]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self.avg_dl)))
                doc_score += idf * ((tf * (self.k1 + 1.0)) / max(1e-9, denom))
            scores[idx] = doc_score
        max_s = float(scores.max()) if len(scores) else 0.0
        return scores / max_s if max_s > 0 else scores

    def query(
        self,
        text: str,
        top_k: int = 3,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Return top-k regulatory controls using 'hybrid', 'tfidf', or 'bm25'."""
        q_vec = self.vectorizer.transform([text])
        tfidf_sims = cosine_similarity(q_vec, self.tfidf_matrix).ravel()
        bm25_sims = self._bm25_scores(text)

        if mode == "tfidf":
            scores = tfidf_sims
        elif mode == "bm25":
            scores = bm25_sims
        else:
            scores = 0.55 * tfidf_sims + 0.45 * bm25_sims

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        q_set = set(_tokenize(text))
        for idx in ranked_indices:
            ctrl = self.controls[int(idx)]
            matched_kw = [kw for kw in ctrl.keywords if any(tok in q_set for tok in _tokenize(kw))]
            results.append(
                {
                    "control_id": ctrl.control_id,
                    "framework": ctrl.framework,
                    "domain": ctrl.domain,
                    "title": ctrl.title,
                    "description": ctrl.description,
                    "relevance_score": round(float(min(1.0, scores[int(idx)])), 4),
                    "matched_keywords": matched_kw[:4],
                    "remediation_steps": ctrl.remediation_steps,
                    "estimated_effort_hours": ctrl.estimated_effort_hours,
                }
            )
        return results

    def map_finding_controls(
        self,
        summary: str,
        reason_codes: list[str] | None = None,
        top_k: int = 2,
    ) -> list[str]:
        """Map a scored audit finding strictly from observable summary & reason codes."""
        query_text = f"{summary} {' '.join(reason_codes or [])}".strip()
        hits = self.query(query_text, top_k=top_k, mode="hybrid")
        return [h["control_id"] for h in hits]


# 24 realistic, paraphrased IT audit queries (avoiding verbatim copying of control titles)
EVAL_QUERIES: list[tuple[str, list[str]]] = [
    ("staff accumulated extra permissions across departments over several months without manager sign-off", ["HKMA-TMG1-4.1", "ISO27001-A5.15", "ISO27001-A5.18"]),
    ("same user can submit a wire transfer and approve the settlement in SWIFT", ["HKMA-TMG1-4.3", "ISO27001-A5.3"]),
    ("engineer pushed firewall and IAM policy edits to live servers without any ServiceNow approval", ["HKMA-TMG1-5.2", "ISO27001-A8.32"]),
    ("large customer identity spreadsheet downloaded at 3am outside normal Hong Kong office hours", ["HKMA-TMG1-6.1", "ISO27001-A8.12"]),
    ("former vendor login remained active for four months and suddenly signed in with password errors", ["HKMA-CRAF-3.2", "ISO27001-A5.18"]),
    ("analyst pasted proprietary client holdings into a public ChatGPT style external API", ["HKMA-TMG1-7.4", "NIST-AI-GOVERN-1.2", "NIST-AI-MAP-1.5"]),
    ("database administrator used permanent root credentials instead of just-in-time vault checkout", ["ISO27001-A8.2"]),
    ("security operations lacked tamper-proof WORM storage to correlate repeated login failures", ["ISO27001-A8.15"]),
    ("internal analytics pipeline transmitted raw customer IDs without masking or encryption", ["ISO27001-A8.24"]),
    ("quarterly check on whether the anomaly model false positive rate is drifting across business units", ["NIST-AI-MEASURE-2.6"]),
    ("escalation path for compliance officers to override or escalate critical AI alerts to CISO", ["NIST-AI-MANAGE-2.2"]),
    ("access matrix does not match actual departmental job responsibilities", ["ISO27001-A5.15", "HKMA-TMG1-4.1"]),
    ("lack of maker-checker dual control in ERP accounting and role approvals", ["ISO27001-A5.3", "HKMA-TMG1-4.3"]),
    ("unregistered external foundation models used without corporate CASB blocking or inventory", ["NIST-AI-GOVERN-1.2", "HKMA-TMG1-7.4"]),
    ("missing watermarking and content inspection on bulk regulatory data downloads", ["ISO27001-A8.12", "HKMA-TMG1-6.1"]),
    ("employee transferred teams six months ago but old portfolio ledger rights were never removed", ["ISO27001-A5.18", "HKMA-TMG1-4.1"]),
    ("direct console hotfix bypassed CI/CD peer review and lacked rollback documentation", ["ISO27001-A8.32", "HKMA-TMG1-5.2"]),
    ("developer holds write access to both code repository and production payment gateway", ["HKMA-TMG1-4.3", "ISO27001-A5.3", "ISO27001-A8.2"]),
    ("sensitive KYC documents fed into third-party AI prompts without PII scrubbing middleware", ["NIST-AI-MAP-1.5", "HKMA-TMG1-7.4"]),
    ("privileged session recording was disabled during emergency IAM console elevation", ["ISO27001-A8.2", "ISO27001-A8.15"]),
    ("automated HR offboarding failed to suspend stale contractor accounts after 30 days", ["HKMA-CRAF-3.2", "ISO27001-A5.18"]),
    ("peer-group baseline alert fatigue requires recalibrating SIEM detection thresholds", ["ISO27001-A8.15", "NIST-AI-MEASURE-2.6"]),
    ("client confidentiality boundaries were not classified before granting AI sandbox read access", ["NIST-AI-MAP-1.5", "ISO27001-A8.24"]),
    ("auditor triage decisions lack immutable logs recording reviewer identity and rationale", ["NIST-AI-MANAGE-2.2", "ISO27001-A8.15"]),
]


def _eval_single_mode(retriever: HybridComplianceRetriever, mode: str) -> dict[str, Any]:
    r1_hits = 0
    r3_hits = 0
    rr_sum = 0.0
    details: list[dict[str, Any]] = []

    for query_text, expected_ids in EVAL_QUERIES:
        preds = [h["control_id"] for h in retriever.query(query_text, top_k=3, mode=mode)]
        hit1 = int(bool(preds) and preds[0] in expected_ids)
        hit3 = int(any(p in expected_ids for p in preds))
        rr = 0.0
        for rank, cid in enumerate(preds, start=1):
            if cid in expected_ids:
                rr = 1.0 / rank
                break
        r1_hits += hit1
        r3_hits += hit3
        rr_sum += rr
        details.append(
            {
                "query": query_text,
                "expected": expected_ids,
                "predicted_top3": preds,
                "hit_at_3": bool(hit3),
            }
        )

    n = len(EVAL_QUERIES)
    return {
        "mode": mode,
        "num_queries": n,
        "recall_at_1": round(r1_hits / n, 4),
        "recall_at_3": round(r3_hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "sample_evaluations": details[:5],
    }


def evaluate_retriever(retriever: HybridComplianceRetriever) -> dict[str, Any]:
    """Run 3-way retrieval ablation (TF-IDF vs BM25 vs Hybrid) on 24 paraphrased audit queries."""
    tfidf_res = _eval_single_mode(retriever, "tfidf")
    bm25_res = _eval_single_mode(retriever, "bm25")
    hybrid_res = _eval_single_mode(retriever, "hybrid")

    ablation = [
        {
            "mode": "TF-IDF Cosine Only",
            "recall_at_1": tfidf_res["recall_at_1"],
            "recall_at_3": tfidf_res["recall_at_3"],
            "mrr": tfidf_res["mrr"],
        },
        {
            "mode": "BM25 Lexical Only",
            "recall_at_1": bm25_res["recall_at_1"],
            "recall_at_3": bm25_res["recall_at_3"],
            "mrr": bm25_res["mrr"],
        },
        {
            "mode": "Hybrid (0.55 TF-IDF + 0.45 BM25)",
            "recall_at_1": hybrid_res["recall_at_1"],
            "recall_at_3": hybrid_res["recall_at_3"],
            "mrr": hybrid_res["mrr"],
        },
    ]

    return {
        "num_queries": hybrid_res["num_queries"],
        "recall_at_1": hybrid_res["recall_at_1"],
        "recall_at_3": hybrid_res["recall_at_3"],
        "mrr": hybrid_res["mrr"],
        "ablation": ablation,
        "sample_evaluations": hybrid_res["sample_evaluations"],
    }
