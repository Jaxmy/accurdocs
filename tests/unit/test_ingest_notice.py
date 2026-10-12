"""入库回执（notice）回归：版本判语 + 自动开启富化 + ETA + 扫描中可提问提示。"""

from policyrag.ingest.pipeline import build_ingest_notice


def _ok(**kw):
    rep = {"status": "ok", "title": "辽宁省电力市场运营规则", "version": "2023",
           "n_formulas": 51, "n_charts": 12, "eta_seconds": 480,
           "warnings": [], "version_judgment": {
               "t0_markers": {"doc_no": "辽发改能源〔2023〕15号", "draft_state": "试行",
                              "version_marker": None, "period": None, "has_marker": True},
               "relation": "new", "t1_confirmed": True}}
    rep.update(kw)
    return rep


def test_notice_new_family():
    s = build_ingest_notice(_ok())
    assert "辽宁省电力市场运营规则" in s
    assert "新文档家族" in s
    assert "标题快判" in s and "全文确判" in s
    assert "已自动开启扫描：公式 51 处、图表 12 处" in s
    assert "预计约 8 分钟" in s
    assert "扫描过程中也可以提问" in s


def test_notice_same_doc_version():
    s = build_ingest_notice(_ok(version_judgment={
        "t0_markers": {"version_marker": "2023年修订", "has_marker": True},
        "relation": "same_doc_version", "near_title": "辽宁省电力市场运营规则（试行）",
        "near_version": "2022", "t1_confirmed": True}))
    assert "同一文档家族（版本演进）" in s
    assert "辽宁省电力市场运营规则（试行）" in s


def test_notice_no_rich_and_noop_and_vision_warn():
    s = build_ingest_notice(_ok(n_formulas=0, n_charts=0, eta_seconds=0))
    assert "无需公式/图表扫描" in s
    s2 = build_ingest_notice({"status": "noop", "title": "T", "version": "v1"})
    assert "跳过重复入库" in s2
    s3 = build_ingest_notice(_ok(warnings=[
        "[vision_unavailable] 检测到复杂图表 12 个，但未通过视觉能力验证——图表数值类问题将拒答。"]))
    assert "视觉" in s3 and "拒答" in s3
    assert build_ingest_notice({"status": "rejected", "reason": "x"}) == ""
