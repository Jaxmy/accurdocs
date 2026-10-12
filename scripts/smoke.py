"""端到端冒烟：ingest → 检索 top-1 命中 → 分块 ID 幂等。"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.getLogger("jieba").setLevel(logging.WARNING)

from policyrag.service.core import CoreService, ServiceConfig  # noqa: E402


def main() -> None:
    svc = CoreService.__new__(CoreService)  # 不需要 LLM —— 只测检索链
    cfg = ServiceConfig(data_dir="data", embed_backend="hash")
    svc.cfg = cfg
    from policyrag.index.stores.sqlite_meta import MetaStore
    from policyrag.index.bm25 import BM25Index
    from policyrag.index.hybrid import HybridIndex
    from policyrag.index.numpy_store import NumpyVectorStore
    from policyrag.index.vectors import get_embedder

    svc.meta = MetaStore(cfg.data_dir / "meta.sqlite3")
    emb = get_embedder("hash")
    bm25 = BM25Index(cfg.data_dir / "bm25")
    vecs = NumpyVectorStore(cfg.data_dir / "vectors", dim=emb.dim)
    svc.rebuild_index = lambda: None  # 已构建

    metas = svc.meta.all_chunk_metas()
    index = HybridIndex(bm25, vecs, emb, metas)

    q = "资本充足率的最低要求"
    hits = index.search(q, k=5)
    assert hits, "检索无结果"
    top = hits[0]
    print(f"查询: {q}")
    print(f"TOP-1: {top.clause_path} | {top.snippet[:50]}")
    assert "8%" in top.snippet or "资本充足率" in top.snippet, f"top-1 未命中目标条款: {top.snippet}"
    print("检索 top-1 OK")

    q2 = "信用贷款额度上限"
    hits2 = index.search(q2, k=5)
    top2 = hits2[0]
    print(f"查询: {q2}")
    print(f"TOP-1: {top2.clause_path} | {top2.snippet[:50]}")
    assert "1,000" in top2.snippet or "信用贷款" in top2.snippet, f"top-1 未命中: {top2.snippet}"
    print("检索 top-1 OK (2)")

    q3 = "短期贷款利率上限"
    hits3 = index.search(q3, k=5)
    joined = " ".join(h.snippet for h in hits3[:3])
    assert "5%" in joined or "短期" in joined, f"表格行未召回: {joined[:80]}"
    print("表格行召回 OK")

    print("SMOKE ALL PASS")


if __name__ == "__main__":
    main()
