"""数字保真 + 引用校验测试（硬闸：不通过必须拒答/删句）。"""

from policyrag.models.answer import Citation
from policyrag.verify.citations import CitationFailure, verify_citations
from policyrag.verify.deterministic import (
    split_sentences,
    strip_failing_sentences,
    verify_numbers,
)

EV = "第十二条 贷款人应确保资本充足率不低于 8%，杠杆率不得超过 6%。2024 年 3 月 1 日起施行。"


def test_numbers_ok_when_verbatim():
    ans = "资本充足率不得低于 8%。[a]"
    r = verify_numbers(ans, [EV])
    assert r.ok, r.failures


def test_numbers_fail_on_fabricated():
    ans = "资本充足率不得低于 10%。[a]"
    r = verify_numbers(ans, [EV])
    assert not r.ok
    assert any(f.canonical == "10%" for f in r.failures)


def test_numbers_fullwidth_and_unit_forms():
    r = verify_numbers("资本充足率不得低于 ８％。", [EV])
    assert r.ok, r.failures
    r2 = verify_numbers("资本充足率不得低于百分之八。", [EV])
    assert r2.ok, r2.failures


def test_numbers_direction_binding():
    ev = "不良贷款率下调 5%。"
    ok = verify_numbers("不良贷款率下调 5%。", [ev])
    assert ok.ok, ok.failures
    bad = verify_numbers("不良贷款率上调 5%。", [ev])
    # 方向词不一致时应有失败记录
    assert bad.failures


def test_date_canonicalization():
    r = verify_numbers("本办法自 2024 年 3 月 1 日起施行。", [EV])
    assert r.ok, r.failures
    r2 = verify_numbers("本办法自 2024-03-01 起施行。", [EV])
    assert r2.ok, r2.failures
    r3 = verify_numbers("本办法自 2024-05-01 起施行。", [EV])
    assert not r3.ok


def test_strip_failing_sentences():
    ans = "第一条结论正确，比率是 8%。第二条编造了 10%。"
    r = verify_numbers(ans, [EV])
    stripped, removed = strip_failing_sentences(ans, r.failures)
    assert removed == 1
    assert "10%" not in stripped
    assert "8%" in stripped


def test_citation_quote_must_hit_verbatim():
    registry = {"abc123abc123abc1": EV}
    retrieved = {"abc123abc123abc1"}
    good = Citation(chunk_id="abc123abc123abc1", quote="资本充足率不低于 8%",
                    doc_title="t", doc_version="v", clause_path="第十二条")
    r = verify_citations([good], registry, retrieved)
    assert r.ok and r.verified_count == 1

    bad = Citation(chunk_id="abc123abc123abc1", quote="资本充足率不低于 12%",
                   doc_title="t", doc_version="v", clause_path="第十二条")
    r2 = verify_citations([bad], registry, retrieved)
    assert not r2.ok


def test_citation_must_be_in_retrieved_set():
    registry = {"abc123abc123abc1": EV, "def456def456def4": "另一段"}
    c = Citation(chunk_id="def456def456def4", quote="另一段",
                 doc_title="t", doc_version="v", clause_path="")
    # 本次检索集合不含 def456 → 引用无效（防编造"库里有但没检索到"）
    r = verify_citations([c], registry, {"abc123abc123abc1"})
    assert not r.ok
    assert "不在本次检索" in r.failures[0].reason


def test_split_sentences_but_clause():
    sents = split_sentences("贷款人应审慎经营。但书情形除外。")
    assert len(sents) >= 1


def test_citation_ids_are_not_numbers():
    """回归：chunk_id 里的 hex 数字段不许被当成数值校验（曾误杀正常答案）。"""
    ev = "资本充足率不得低于 8%。"
    # 带括号引用标记
    ans = "资本充足率不得低于 8%。[9ce4136db20f1dff]"
    r = verify_numbers(ans, [ev])
    assert r.ok, r.failures
    assert r.checked_spans == 1  # 只有 8%，ID 不算
    # 裸 hex ID
    ans2 = "资本充足率不得低于 8%。9ce4136db20f1dff"
    r2 = verify_numbers(ans2, [ev])
    assert r2.ok, r2.failures
    # 真实数字仍然要校验
    ans3 = "资本充足率不得低于 12%。[9ce4136db20f1dff]"
    r3 = verify_numbers(ans3, [ev])
    assert not r3.ok
    # 纯数字长串（如编号 20241231）不算 hex ID 保护区，仍校验
    ans4 = "文号 20241231 的规定，资本充足率不得低于 8%。"
    r4 = verify_numbers(ans4, [ev])
    assert not r4.ok  # 20241231 未在 evidence 中


def test_question_numbers_are_user_premises():
    """R2-07：问题自带日期回述不作编造（进溯源白名单）。"""
    q = "产业链调研中，2011年11月17日在杭州调研了哪两家公司？"
    ev = "2011年11月17日，杭州：淘宝商城、一淘网。"
    ans = "2011年11月17日在杭州调研的是淘宝商城和一淘网。"
    assert verify_numbers(ans, [ev], q).ok
    # 证据缺失日期时，问题日期仍是合法前提
    ev2 = "调研对象：淘宝商城、一淘网。"
    assert verify_numbers(ans, [ev2], q).ok, verify_numbers(ans, [ev2], q).failures
    # 但答案编造的其他数值仍拦
    ans2 = "2011年11月17日调研了37家公司。"
    r = verify_numbers(ans2, [ev2], q)
    assert not r.ok and any(f.canonical.startswith("37") for f in r.failures)


def test_strip_keeps_sentence_with_hits():
    """R2-07：句内有命中数值只剥失配 span，不整句删（金提示不被误删）。"""
    ans = "经营主体停牌情形包括连续12个月未交易和违规两类，另有99种其他情况。"
    ev = "连续12个月未进行实际交易的将被停牌。"
    r = verify_numbers(ans, [ev])
    assert not r.ok
    stripped, removed = strip_failing_sentences(ans, r.failures, r.hit_ranges)
    assert "12个月" in stripped          # 命中数值保住
    assert "99" not in stripped           # 失配 span 被剥
    assert removed == 0                   # 句子未整删


def test_strip_deletes_sentence_without_hits():
    ans = "第一句命中8%。第二句编造了10%和20%。"
    ev = "资本充足率不低于 8%。"
    r = verify_numbers(ans, [ev])
    stripped, removed = strip_failing_sentences(ans, r.failures, r.hit_ranges)
    assert removed == 1
    assert "8%" in stripped and "10%" not in stripped and "20%" not in stripped


def test_citation_snap_to_source_window():
    """R2-10 F4b：引文轻微失真 → 快照吸附回原文句窗（唯一最佳+数值同一性守卫）。"""
    from policyrag.verify.citations import verify_citations as vc
    chunk = "（第 38 页表）沃尔玛物流配送中心运作模式及核心功能（Hub-and-Spoke，轴辐模式）。入站物流验收后配货包装。"
    registry = {"cd104fb414dd57e8": chunk}
    retrieved = {"cd104fb414dd57e8"}
    # 模型把断行/标点清洗后引用（非逐字）
    c = Citation(chunk_id="cd104fb414dd57e8",
                 quote="沃尔玛物流配送中心运作模式及核心功能(Hub-and-Spoke 轴辐模式)",
                 doc_title="t", doc_version="v", clause_path="")
    r = vc([c], registry, retrieved)
    assert r.ok, r.failures
    assert r.snapped_count == 1
    assert "轴辐模式" in c.quote            # 吸附后 quote 即原文句窗


def test_citation_snap_rejects_numeric_mismatch():
    """数值同一性守卫：引文数值与候选句窗不符 → 不吸，仍失败。"""
    from policyrag.verify.citations import verify_citations as vc
    chunk = "沃尔玛收入年复合增长6%，净利润增长8%。另一段无关文本在此。"
    registry = {"aaaaaaaaaaaaaaaa": chunk}
    c = Citation(chunk_id="aaaaaaaaaaaaaaaa",
                 quote="沃尔玛收入年复合增长9%，净利润增长8%。",  # 9% 系编造
                 doc_title="t", doc_version="v", clause_path="")
    r = vc([c], registry, {"aaaaaaaaaaaaaaaa"})
    assert not r.ok and r.snapped_count == 0


def test_demote_unmatched_quotes_last_resort():
    """R2-10 F4c：失配引文降级转述——剥引号声明，不再声称逐字。"""
    from policyrag.verify.citations import demote_unmatched_quotes
    c = Citation(chunk_id="bbbbbbbbbbbbbbbb", quote="完全对不上的引文",
                 doc_title="t", doc_version="v", clause_path="")
    fails = [CitationFailure("bbbbbbbbbbbbbbbb",
                             "quote 未逐字命中 chunk 原文（疑似引用幻觉）", "完全对不上的引文")]
    hard = demote_unmatched_quotes([c], fails, {"bbbbbbbbbbbbbbbb": "另一段无关原文。"})
    assert not hard
    assert c.demoted and c.quote == "" and not c.verified


def test_demote_refuses_fabricated_numbers_in_quote():
    """红线：引文携带编造数值 → 不降级，留硬失败拒答。"""
    from policyrag.verify.citations import demote_unmatched_quotes
    c = Citation(chunk_id="cccccccccccccccc", quote="资本充足率不得低于 15%",
                 doc_title="t", doc_version="v", clause_path="")
    fails = [CitationFailure("cccccccccccccccc",
                             "quote 未逐字命中 chunk 原文（疑似引用幻觉）", "资本充足率不得低于 15%")]
    hard = demote_unmatched_quotes([c], fails, {"cccccccccccccccc": EV})
    assert hard and not c.demoted          # 15% 无出处 → 硬拒
    assert c.quote                          # 引文保留供错误报告
