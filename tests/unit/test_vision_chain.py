"""视觉链路回归：能力探测（共享键场景）、双源分层、入库回报、图表转写分层入证。"""

import json
import sys
import types

from policyrag.llm.vision import needs_vision_hint, tier_chart_values, vision_fingerprint


class TestTierChartValues:
    def test_dual_source_agree_strong(self):
        vlm = [{"series": "零售", "axis": "2010", "value": "38,877.25"},
               {"series": "电商", "axis": "2010", "value": "1,665.13"}]
        ocr = ["2010年美国零售业规模 38,877.25 亿美元", "电子商务 1665.13"]
        strong, weak = tier_chart_values(vlm, ocr)
        assert [v["value"] for v in strong] == ["38,877.25", "1,665.13"]
        assert weak == []

    def test_vlm_only_value_is_weak(self):
        """VLM 独有读数（OCR 未见）= 弱证据，不得进保真比对集（防误读自我印证）。"""
        vlm = [{"series": "沃尔玛", "axis": "2010", "value": "2600"}]
        ocr = ["图表标题：规模比较"]
        strong, weak = tier_chart_values(vlm, ocr)
        assert strong == [] and len(weak) == 1

    def test_boundary_no_false_hit(self):
        vlm = [{"series": "x", "axis": "y", "value": "877.25"}]
        ocr = ["总量 38877.25"]   # 子串不得误判双源一致
        strong, weak = tier_chart_values(vlm, ocr)
        assert strong == [] and len(weak) == 1

    def test_empty(self):
        assert tier_chart_values([], []) == ([], [])


class TestNeedsVisionHint:
    def test_research_report_shape(self):
        assert needs_vision_hint(62, 400) is True
        assert needs_vision_hint(2, 6) is True
        assert needs_vision_hint(0, 0) is False
        assert needs_vision_hint(3, 1) is False


class TestProbe:
    def test_probe_mocked_and_cached(self, tmp_path, monkeypatch):
        from policyrag.llm import vision

        class _Res:
            text = "7"
            parsed = None
            usage = None
            model = "mock-vision"

        class _LLM:
            name = "mock"
            model = "mock-vision"
            caps = type("C", (), {"structured_output": False, "json_mode": True})()

            def complete(self, messages, **kw):
                assert messages[0].images, "探测必须携带测试图"
                return _Res()

        monkeypatch.setattr(vision, "make_vision_llm", lambda: _LLM())
        monkeypatch.setenv("POLICYRAG_DATA", str(tmp_path))
        # probe 缓存写 PROJECT_ROOT/data —— 指到 tmp 由 settings.PROJECT_ROOT 决定，
        # 这里直接断言返回值；缓存命中逻辑用 force=False 二调验证
        r1 = vision.probe_vision(force=True)
        assert r1["capable"] is True and r1["model"] == "mock-vision"

        def _boom(*a, **k):
            raise AssertionError("force=False 应命中缓存，不再发起探测")

        monkeypatch.setattr(vision, "make_vision_llm", _boom)
        r2 = vision.probe_vision(force=False)
        assert r2["capable"] is True

    def test_fingerprint_changes_with_key(self, monkeypatch):
        monkeypatch.setenv("POLICYRAG_VISION_API_KEY", "sk-a")
        f1 = vision_fingerprint()
        monkeypatch.setenv("POLICYRAG_VISION_API_KEY", "sk-b")
        assert vision_fingerprint() != f1


class TestChartProcessTiering:
    def test_chart_body_strong_vs_weak(self, tmp_path, monkeypatch):
        """强证据值入正文数据表；弱值只进 meta.data_table（不进保真比对）。"""
        from policyrag.ingest.enrichment import RichWorker
        from policyrag.index.stores.sqlite_meta import MetaStore

        from policyrag.models.ids import make_rich_chunk_id

        m = MetaStore(str(tmp_path / "meta.sqlite3"))
        m.upsert_document("研报", "v1", "研报")
        img = tmp_path / "g1.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        cid = make_rich_chunk_id("研报", "v1", "chart", "g_001_01")
        m.upsert_chunk(cid, "研报@v1", "图表/g_001_01", "第1页 图表g_001_01",
                       "（第1页图表，图内文字转写进行中）", "占位", "chart", page=1,
                       meta={"title": "规模比较", "image": str(img), "pending": True})
        m.commit()

        import policyrag.ingest.native_worker as NW
        monkeypatch.setattr(NW, "ocr_isolated",
                            lambda img, timeout=120: ["38877.25", "标题"])

        from policyrag.llm import vision as V
        monkeypatch.setattr(V, "probe_vision", lambda force=False: {"capable": True})
        monkeypatch.setattr(V, "make_vision_llm", lambda: object())
        monkeypatch.setattr(V, "transcribe_chart", lambda llm, b: {
            "title": "规模比较", "chart_type": "bar",
            "values": [{"series": "零售", "axis": "2010", "value": "38877.25"},
                       {"series": "电商", "axis": "2010", "value": "1665.13"}]})

        w = RichWorker(m)
        w._process_charts("j1", "研报@v1", {
            "file_hash": "fhx", "chart_local_ids": ["g_001_01"],
            "doc_id": "研报", "version": "v1", "use_vision": True})

        row = m.get_chunk(cid)
        assert "【数据表】" in row["text"] and "38877.25" in row["text"]
        assert "1665.13" not in row["text"], "弱值不得进证据文本"
        meta = json.loads(row["meta_json"])
        assert "1665.13" in meta["data_table"], "弱值保留在展示数据表"
        assert meta["transcribed_by"] == "vlm+ocr" and meta["confidence"] == "high"
