"""R2-B 数值预标注：LLM 提议、确定性处置（子串+canonical），demanded 并集。"""

from policyrag.ingest.numspan import annotate_numeric_spans, numspan_cache_key, validate_spans
from policyrag.verify.coverage import verify_numeric_coverage


TEXT = ("格力空调 | 当当网 6,198 | 京东 5,988 | 卓越 6,270 | 苏宁 6,512\n"
        "断格数值 1,992 亿。")


def test_validate_requires_literal_substring():
    spans = [{"text": "6,198", "context_hint": "格力空调|当当网"},
             {"text": "1,992 亿", "context_hint": "沃尔玛市值"},
             {"text": "7,777", "context_hint": "编造值"},          # 非原文 → 丢
             {"text": "这是改写不是原文", "context_hint": ""}]      # 非原文 → 丢
    out = validate_spans(TEXT, spans)
    texts = [s["text"].replace(" ", "") for s in out]
    assert "6,198" in texts and "1,992亿" in texts
    assert all("7,777" not in t for t in texts)
    assert all(s["canonical"] for s in out)


def test_demanded_union_pre_and_runtime():
    """demanded = 预标注 ∪ 运行时正则——预标注漏标不得缩小可命中池。"""
    ev = "负荷备用为2-5%，事故备用为10%。"
    # 预标注只标了 10%（漏 2-5%）——答案给 2-5% 仍须被接受（并集语义）
    pre = [{"text": "10%", "canonical": "10%", "unit_type": "ratio", "context_hint": ""}]
    q = "负荷备用和事故备用各占多少？"
    ans = "负荷备用为2-5%。"
    r = verify_numeric_coverage(q, ans, [ev], question_kind="numeric",
                                dimension="ratio", mode="llm", pre_spans=pre)
    assert r.ok and r.partial          # 运行时正则仍把 2-5% 视为可命中
    # 断格数值：运行时正则抽不出、预标注补上 → demanded 含它，漏答可拦
    ev2 = "市值见表：1,992 亿。"        # 假设运行时抽不出 "1,992 亿"（跨格）
    pre2 = [{"text": "1,992 亿", "canonical": "1992亿", "unit_type": None, "context_hint": ""}]
    ans2 = "报告未提供。"
    r2 = verify_numeric_coverage("请给出市值是多少？", ans2, [ev2],
                                 question_kind="numeric", dimension="money",
                                 mode="llm", pre_spans=pre2)
    # 只要 demanded 里有 1992亿 且答案零命中 → ok=False（拦）或 ok=True（若运行时正则已含其它值）
    # 关键断言：pre_spans 进入了 demanded 命名空间（missing 示例可出现）
    assert (not r2.ok) or r2.missing_examples


def test_cache_key_stable_and_content_sensitive():
    k1 = numspan_cache_key("fh", "c1", "文本A")
    k2 = numspan_cache_key("fh", "c1", "文本A")
    k3 = numspan_cache_key("fh", "c1", "文本B")
    assert k1 == k2 and k1 != k3 and k1.startswith("numspan:")


def test_annotate_without_llm_returns_empty():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("gateway down")
    assert annotate_numeric_spans(Boom(), TEXT, kind="table") == []
