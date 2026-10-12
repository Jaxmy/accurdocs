"""混合检索：BM25 + 稠密向量 + RRF 融合（计划 §三 index/hybrid）。"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.retrieval import Hit
from .bm25 import BM25Index
from .numpy_store import NumpyVectorStore
from .vectors import Embedder

__all__ = ["HybridIndex", "RRF_K"]

RRF_K = 60  # RRF 平滑常数，经验值


@dataclass
class HybridIndex:
    bm25: BM25Index
    vectors: NumpyVectorStore
    embedder: Embedder
    # registry: chunk_id -> chunk 元信息（snippet 生成用）
    metas: dict[str, dict]

    def search(
        self,
        query: str,
        k: int = 8,
        candidate_k: int = 50,
        terms: list[str] | None = None,
        numeric_anchors: list[str] | None = None,
    ) -> list[Hit]:
        """BM25(k) ∪ dense(k) → RRF 融合 → top-k。

        terms/numeric_anchors：directive 提供的精确锚点——命中加分（不做过滤，
        过滤留给 temporal/metadata 层；这里只做加权，防召回坍塌）。
        """
        bm25_hits = self.bm25.search(query, k=candidate_k)
        qvec = self.embedder.embed_query(query)
        dense_hits = self.vectors.search(qvec, k=candidate_k)

        # RRF 融合
        scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        for rank, (cid, s) in enumerate(bm25_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            bm25_scores[cid] = s
        for rank, (cid, s) in enumerate(dense_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            dense_scores[cid] = s

        # 锚点加权：命中条款号/术语/数值锚点的 chunk 提升
        # 条款形 anchor 走边界感知规范匹配（"12条"不误中"112条"，跨形态互通）
        from ..models.clause_norm import canon_clause_ref, clause_ref_hits

        anchors = set(terms or []) | set(numeric_anchors or [])
        if anchors:
            for cid in scores:
                meta = self.metas.get(cid, {})
                text = (meta.get("text_norm") or "") + " " + (meta.get("breadcrumb") or "")
                bonus = 0.0
                for a in anchors:
                    if not a:
                        continue
                    hit = clause_ref_hits(a, text) if canon_clause_ref(a) else (a in text)
                    if hit:
                        bonus += 0.002
                scores[cid] += bonus

        ranked_all = sorted(scores.items(), key=lambda kv: -kv[1])
        # BM25 保护位：稀疏精确命中的前 3 名必进 top-k（防 dense 噪声淹没——
        # 实测 BM25 第 1 名被 50 个 dense 公式块 RRF 稀释挤出证据窗）
        protected = [cid for cid, _ in bm25_hits[:3]]
        seen_p = set(protected)
        fused = [cid for cid, _ in ranked_all if cid not in seen_p]
        final_ids = (protected + fused)[: max(k, len(protected))]
        score_of = dict(ranked_all)
        ranked = [(cid, score_of.get(cid, 0.0)) for cid in final_ids]
        out: list[Hit] = []
        for cid, score in ranked:
            meta = self.metas.get(cid, {})
            out.append(Hit(
                chunk_id=cid,
                score=score,
                bm25_score=bm25_scores.get(cid, 0.0),
                dense_score=dense_scores.get(cid, 0.0),
                snippet=(meta.get("text") or "")[:160],
                doc_title=meta.get("doc_title", ""),
                doc_version=meta.get("version", ""),
                clause_path=meta.get("breadcrumb", ""),
            ))
        return out
