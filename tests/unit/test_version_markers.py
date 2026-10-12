"""两级版本判定（T0 标题快判 + T1 全文确判）回归。

用户决策：版本相似度判断先按标题索引判版本（政策版本号/文号/稿态，财报财年季度），
全文指纹确判；同基名+版本标记强信号 → 大改版不拆成两文档。
"""

from policyrag.ingest.identity import (
    base_title, decide_identity, extract_version_markers,
)


def test_policy_markers():
    mk = extract_version_markers("辽宁省电力市场运营规则（试行）\n辽发改能源〔2023〕15号\n2023年修订")
    assert mk["doc_no"] == "辽发改能源〔2023〕15号"
    assert mk["draft_state"] == "试行"
    assert mk["version_marker"] and "修订" in mk["version_marker"]
    assert mk["has_marker"]


def test_report_period_markers():
    assert extract_version_markers("XX公司2024年三季度报告")["period"] == "2024Q3"
    assert extract_version_markers("XX公司2024Q3报告")["period"] == "2024Q3"
    assert extract_version_markers("XX证券2024年半年度报告")["period"] == "2024H1"
    assert extract_version_markers("XX公司2024年度报告")["period"] == "2024年度"
    assert extract_version_markers("华创证券-寻找无边界扩张的行业")["has_marker"] is False


def test_base_title_strips_markers():
    assert base_title("辽宁省电力市场运营规则（试行）") == \
        base_title("辽宁省电力市场运营规则")
    assert base_title("XX公司2024年三季度报告") == base_title("XX公司")
    assert base_title("辽宁省电力市场运营规则") != base_title("南方电力市场运营规则")


def _row(doc_key, title, group=None, seq=1, chash="", simhash="0", meta=None):
    import json as _j
    return {"doc_key": doc_key, "doc_id": doc_key.split("@")[0],
            "version": doc_key.split("@")[-1], "title": title,
            "doc_group": group or doc_key.split("@")[0], "version_seq": seq,
            "content_hash": chash, "simhash": simhash,
            "meta_json": _j.dumps(meta or {})}


def test_big_revision_joins_group_via_markers():
    """大改版：同基名+不同版本标记 → 同族（内容大变也归组，不再拆两文档）。"""
    existing = [_row("规则@2022", "辽宁省电力市场运营规则（试行）",
                     meta={"version_markers": {"draft_state": "试行", "has_marker": True}})]
    ident = decide_identity(
        doc_id="规则", version="2023", title="辽宁省电力市场运营规则",
        doc_no=None, raw_text="辽宁省电力市场运营规则\n正式稿\n全文大改……" * 200,
        existing=existing)
    assert ident.doc_group == "规则"
    assert ident.relation == "same_doc_version"
    assert ident.version_seq == 2
    assert ident.markers and ident.markers["has_marker"]


def test_no_marker_title_alone_not_enough():
    """双侧无版本标记时标题相似不并组（单靠标题不成立红线不破）。"""
    existing = [_row("规则@v1", "辽宁省电力市场运营规则",
                     chash="deadbeef", simhash="1")]
    ident = decide_identity(
        doc_id="规则2", version="v1", title="辽宁省电力市场运营规则",
        doc_no=None, raw_text="完全不同的正文内容，没有任何版本标记。" * 100,
        existing=existing)
    assert ident.relation == "new"
    assert ident.doc_group == "规则2"


def test_fiscal_year_versions_group():
    existing = [_row("年报@2023年度", "XX公司2023年度报告",
                     meta={"version_markers": {"period": "2023年度", "has_marker": True}})]
    ident = decide_identity(
        doc_id="年报", version="2024Q3", title="XX公司2024年三季度报告",
        doc_no=None, raw_text="XX公司2024年三季度报告\n业绩大幅变化……" * 200,
        existing=existing)
    assert ident.doc_group == "年报"
    assert ident.relation == "same_doc_version"
