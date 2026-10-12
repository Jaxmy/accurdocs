"""M0 现网修复回归：替换语义 / worker 回调 / 重启续跑 / tokenizer 版本 / 图表异步占位。"""

import json
import time
import uuid

from policyrag.index.bm25 import BM25Index
from policyrag.index.stores.sqlite_meta import MetaStore
from policyrag.index.tokenizer_zh import TOKENIZER_VERSION


def _tmp():
    return f"tests/.tmp/{uuid.uuid4().hex[:8]}"


class TestReplaceChunks:
    def test_stale_chunks_removed_on_reingest(self):
        tmp = _tmp()
        m = MetaStore(f"{tmp}/meta.sqlite3")
        m.upsert_document("doc", "v1", "标题")
        for i in range(3):
            m.upsert_chunk(f"old{i}", "doc@v1", f"p{i}", f"b{i}", "文本", "文本", "text")
        m.commit()
        keep = {"new0", "new1"}
        for i, cid in enumerate(sorted(keep)):
            m.upsert_chunk(cid, "doc@v1", f"n{i}", f"b{i}", "新文本", "新文本", "text")
        n = m.replace_document_chunks("doc@v1", keep)
        assert n == 3
        left = {r["chunk_id"] for r in m.chunks_of_doc("doc@v1")}
        assert left == keep

    def test_replace_noop_when_same(self):
        tmp = _tmp()
        m = MetaStore(f"{tmp}/meta.sqlite3")
        m.upsert_document("doc", "v1", "标题")
        m.upsert_chunk("a", "doc@v1", "p", "b", "t", "t", "text")
        m.commit()
        assert m.replace_document_chunks("doc@v1", {"a"}) == 0


class TestWorkerFixes:
    def test_get_worker_updates_callback_and_meta(self):
        from policyrag.ingest.enrichment import get_worker
        tmp = _tmp()
        m1 = MetaStore(f"{tmp}/m1.sqlite3")
        m2 = MetaStore(f"{tmp}/m2.sqlite3")
        calls = []

        w1 = get_worker(m1)
        w2 = get_worker(m2, on_index_refresh=lambda: calls.append(1))
        assert w1 is w2
        assert w2.meta is m2, "meta 应更新到当前服务实例（不被首调用方冻结）"
        w2.on_index_refresh()
        assert calls == [1], "on_index_refresh 应更新注册（历史 bug：首调用方冻结）"

    def test_resume_pending_requeues(self):
        from policyrag.ingest.enrichment import RichWorker
        tmp = _tmp()
        m = MetaStore(f"{tmp}/meta.sqlite3")
        m.job_upsert("rj_run", "doc@v1", "formula", {"pages": [1]}, status="running", progress=0.4)
        m.job_upsert("rj_que", "doc@v1", "chart", {"chart_local_ids": ["c"]}, status="queued")
        m.job_upsert("rj_done", "doc@v1", "chart", {}, status="done")
        w = RichWorker(m)
        w._ensure_thread = lambda: None   # 不起线程，只验入队与状态翻转
        n = w.resume_pending()
        assert n == 2
        assert m.job_get("rj_run")["status"] == "queued"
        assert "resumed" in (m.job_get("rj_run")["note"] or "")
        assert m.job_get("rj_done")["status"] == "done"


class TestTokenizerVersion:
    def _build(self, root):
        b = BM25Index(root)
        b.build(["c1", "c2"], ["资本充足率不得低于 8%", "杠杆率不得超过 6%"])
        return b

    def test_mismatch_with_texts_retokenizes(self):
        from policyrag.index.tokenizer_zh import tokenize
        tmp = _tmp()
        self._build(f"{tmp}/bm25")
        p = f"{tmp}/bm25/corpus.json"
        data = json.loads(open(p, encoding="utf-8").read())
        data["tokenizer_version"] = "0"     # 人为过期
        data["tokenized"] = [["坏词元"]]     # 旧分词必被打散重建
        open(p, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False))
        b2 = BM25Index(f"{tmp}/bm25")
        b2._open()   # 触发版本检查与重分词
        assert not b2.stale
        data2 = json.loads(open(p, encoding="utf-8").read())
        assert data2["tokenizer_version"] == TOKENIZER_VERSION
        # 重分词结果应等于现规则 tokenize(texts)，旧"坏词元"不残留
        assert data2["tokenized"] == [tokenize(t) for t in data2["texts"]]
        assert all("坏词元" not in toks for toks in data2["tokenized"])

    def test_mismatch_without_texts_marks_stale(self):
        tmp = _tmp()
        self._build(f"{tmp}/bm25")
        p = f"{tmp}/bm25/corpus.json"
        data = json.loads(open(p, encoding="utf-8").read())
        del data["texts"]
        data["tokenizer_version"] = "0"
        open(p, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False))
        b2 = BM25Index(f"{tmp}/bm25")
        b2._open()
        assert b2.stale, "无 texts 无法自愈 → 标 stale 由 CoreService 重建"


class TestChartPlaceholderAsync:
    def test_ingest_chart_placeholder_no_ocr(self):
        """图表入库只截图占位（pending），RapidOCR 全程不被 import/构造。"""
        import sys
        import types

        tmp = _tmp()
        # 注入会炸的 rapidocr——占位路径若偷偷跑 OCR 会立刻失败
        fake = types.ModuleType("rapidocr")

        class Boom:
            def __init__(self):
                raise AssertionError("ingest 不应构造 RapidOCR")

        fake.RapidOCR = Boom
        sys.modules["rapidocr"] = fake

        import pymupdf
        from PIL import Image

        img_path = f"{tmp}/img.png"
        import os
        os.makedirs(tmp, exist_ok=True)
        Image.new("RGB", (150, 150), (200, 100, 50)).save(img_path)
        pdf_path = f"{tmp}/t.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=500)
        body = ("Figure 1: sample chart of policy data. " * 8) + (
            "This document contains a demonstration chart used for testing "
            "the asynchronous enrichment pipeline of the policyrag system.")
        page.insert_textbox(pymupdf.Rect(20, 200, 380, 480), body)
        page.insert_image(pymupdf.Rect(20, 20, 170, 170), filename=img_path)
        doc.save(pdf_path)
        doc.close()

        submitted = []

        class StubWorker:
            def submit(self, *a, **kw):
                submitted.append((a, kw))
                return f"rj_{len(submitted)}"

            def run_inline(self, job_id):
                pass

        import policyrag.ingest.pipeline as pl
        from policyrag.ingest.enrichment import get_worker
        from policyrag.ingest.pipeline import ingest_file

        orig = get_worker
        try:
            import policyrag.ingest.enrichment as em
            em.get_worker = lambda meta, on_index_refresh=None: StubWorker()
            m = MetaStore(f"{tmp}/meta.sqlite3")
            t0 = time.time()
            rep = ingest_file(pl.Path(pdf_path), m, enrich_formulas=False,
                              enrich_charts=True)
            dt = time.time() - t0
        finally:
            em.get_worker = orig
            sys.modules.pop("rapidocr", None)

        assert rep["status"] == "ok"
        assert rep["n_charts"] >= 1
        assert dt < 30, f"入库应秒级，实测 {dt:.1f}s"
        # 占位块 pending=True
        charts = [r for r in m.chunks_of_doc(rep["doc_key"]) if r["kind"] == "chart"]
        assert charts and all('"pending":true' in (c["meta_json"] or "").replace(" ", "") for c in charts)
        # 图表任务已入队（后台 OCR）
        assert submitted, "图表富化任务应提交后台队列"
