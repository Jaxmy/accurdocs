"""富化引擎测试：漏斗 / 稳定 ID / 缓存 / 按需深转。"""

from __future__ import annotations

import json
import uuid

from policyrag.ingest.enrichment import RichWorker, file_hash_of, triage_pdf
from policyrag.models.ids import make_rich_chunk_id


GOLDEN_PDF = "data/formula_eval/golden_pages.pdf"


def test_rich_chunk_id_stable_across_content_upgrade():
    """富化块身份=版面位置：转写升级后 ID 不变（引用/缓存不失效）。"""
    a = make_rich_chunk_id("doc", "v1", "formula", "f_001_01")
    b = make_rich_chunk_id("doc", "v1", "formula", "f_001_01")
    assert a == b
    assert a != make_rich_chunk_id("doc", "v1", "formula", "f_001_02")
    assert a != make_rich_chunk_id("doc", "v2", "formula", "f_001_01")


def test_triage_finds_formula_pages():
    import os
    if not os.path.exists(GOLDEN_PDF):
        return
    t = triage_pdf(GOLDEN_PDF)
    assert len(t["formula_pages"]) >= 5, t
    assert all(p >= 1 for p in t["formula_pages"])


def test_worker_cache_and_upgrade():
    """缓存命中路径：预置 cache 后 run_inline 不跑 MinerU 也能升级占位块。"""
    import os
    import tempfile
    from pathlib import Path

    if not os.path.exists(GOLDEN_PDF):
        return
    tmp = Path(tempfile.mkdtemp(prefix="enrich_test_"))
    from policyrag.index.stores.sqlite_meta import MetaStore
    meta = MetaStore(tmp / "meta.sqlite3")
    doc_id, version = "富化测试", "v1"
    doc_key = f"{doc_id}@{version}"
    fhash = file_hash_of(GOLDEN_PDF)
    meta.upsert_document(doc_id, version, "富化测试", effective_date="2026-01-01",
                         meta={"source_file": GOLDEN_PDF})
    # 预置第 1 页缓存（模拟 MinerU 已跑过）
    meta.enrich_cache_put(fhash, 1, "formula", {"records": [
        {"latex": "R = Max[0]", "symbols": [], "context_before": "计算公式如下", "fixes": []}
    ]})
    meta.commit()

    # 占位块
    cid = make_rich_chunk_id(doc_id, version, "formula", "f_001_01")
    meta.upsert_chunk(chunk_id=cid, doc_key=doc_key, path_key="公式/f_001_01",
                      breadcrumb="第1页 公式f_001_01", text="占位", text_norm="占位",
                      kind="formula", page=1, meta={"pending": True})
    meta.commit()

    worker = RichWorker(meta)
    jid = worker.submit(doc_key, doc_id, version, GOLDEN_PDF, "formula", pages=[1, 2])
    worker.run_inline(jid)
    job = meta.job_get(jid)
    assert job["status"] == "done", job
    row = meta.get_chunk(cid)
    assert row and "Max" in row["text"], row
    extra = json.loads(row["meta_json"])
    assert extra.get("pending") is False
    # 第 2 页无缓存 → MinerU 跑过则有记录或空记录，但任务应完成
    assert job["progress"] == 1.0
