"""公式富化机制回归：MinerU 失败不毒化缓存、占位块 OCR 兜底升级（#1 转写卡占位）。"""

import sys
import types


class _FakeOCRResult:
    txts = ["R阻塞 = ∑ Q × P"]


class _FakeRapidOCR:
    def __call__(self, img):
        return _FakeOCRResult()


def test_placeholder_fallback_and_no_poison_cache(tmp_path, monkeypatch):
    from policyrag.ingest.enrichment import RichWorker
    from policyrag.index.stores.sqlite_meta import MetaStore

    tmp = tmp_path
    m = MetaStore(str(tmp / "meta.sqlite3"))
    doc_key = "测试文档@v1"
    m.upsert_document("测试文档", "v1", "测试文档")
    img = tmp / "f_001_01.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")   # 伪图（FakeOCR 不读内容）
    placeholder = "（第1页公式，原文见附图，文本转写进行中）"
    m.upsert_chunk("fmid", doc_key, "公式/f_001_01", "第1页 公式f_001_01",
                   placeholder, placeholder, "formula", page=1,
                   meta={"latex": "", "image": str(img), "pending": True})
    m.commit()
    m.enrich_cache_put("fh", 1, "formula", {"records": []})   # 毒化缓存
    m.commit()

    import policyrag.ingest.formulas as F
    monkeypatch.setattr(F, "run_mineru", lambda *a, **k: None)   # MinerU 失败
    import policyrag.ingest.native_worker as NW
    monkeypatch.setattr(NW, "ocr_isolated", lambda img, timeout=120: ["R阻塞 = ∑ Q × P"])

    import pymupdf
    pdf = tmp / "doc.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(str(pdf))
    d.close()

    w = RichWorker(m)
    w._process_formulas("job1", doc_key, {
        "file_hash": "fh", "pages": [1], "doc_id": "测试文档", "version": "v1",
        "file_path": str(pdf)})

    row = m.get_chunk("fmid")
    assert "转写进行中" not in row["text"], "占位文案不得残留"
    assert "【公式·OCR】" in row["text"], row["text"]
    import json
    meta = json.loads(row["meta_json"])
    assert meta["transcribed_by"] == "ocr"
    assert meta["pending"] is False


def test_empty_cache_retried_when_placeholder_exists(tmp_path, monkeypatch):
    """空 records 缓存 + 页内仍有占位块 → 视为未命中（MinerU 可重试）。"""
    from policyrag.ingest.enrichment import RichWorker
    from policyrag.index.stores.sqlite_meta import MetaStore

    tmp = tmp_path
    m = MetaStore(str(tmp / "meta.sqlite3"))
    doc_key = "测试文档B@v1"
    m.upsert_document("测试文档B", "v1", "测试文档B")
    placeholder = "（第2页公式，原文见附图，文本转写进行中）"
    m.upsert_chunk("fmidB", doc_key, "公式/f_002_01", "第2页 公式f_002_01",
                   placeholder, placeholder, "formula", page=2,
                   meta={"image": None, "pending": True})
    m.commit()
    m.enrich_cache_put("fh2", 2, "formula", {"records": []})
    m.commit()

    calls = {}

    def fake_mineru(*a, **k):
        calls["n"] = calls.get("n", 0) + 1
        return None

    import policyrag.ingest.formulas as F
    monkeypatch.setattr(F, "run_mineru", fake_mineru)
    import policyrag.ingest.native_worker as NW
    monkeypatch.setattr(NW, "ocr_isolated", lambda img, timeout=120: ["R阻塞"])

    import pymupdf
    pdf = tmp / "doc2.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(str(pdf))
    d.close()

    w = RichWorker(m)
    w._process_formulas("job2", doc_key, {
        "file_hash": "fh2", "pages": [2], "doc_id": "测试文档B", "version": "v1",
        "file_path": str(pdf)})
    assert calls.get("n") == 1, "占位块存在时空缓存必须触发 MinerU 重试"
