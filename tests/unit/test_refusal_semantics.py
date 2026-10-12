"""拒答语义回归（#7）：声明式非答案必拒、拒答话术不暴露内部闸名。"""

from policyrag.models.answer import AnswerStatus, VerifyReport
from policyrag.verify.policy import decide_status


def test_disclaimer_answer_forced_refused():
    """'现有证据均与问题无关'式声明不得判 answered（SA08 曾误判）。"""
    ans = ("现有证据仅涉及绿色债券募集资金用途与碳减排支持工具利率、通用仓库需求面积、"
           "物流库房布局以及辽宁电力市场交易电价与出清规则，均与碳配额价格和套期保值无关。")
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=True, numbers_ok=True),
                            5, 3, answer_text=ans)
    assert st == AnswerStatus.REFUSED
    assert ref and ref.reason == "off_topic_evidence"
    assert "无关" in ref.detail and "知识库" in ref.detail


def test_long_multisentence_disclaimer_refused():
    """MCP 版长声明（多句式、句号截断窗口）曾绕过正则——#7 残洞回归。"""
    ans = ("证据仅涉及绿色债券募集资金用途与碳减排支持工具利率、通用仓库需求面积，以及辽宁电力市场"
           "交易电价与出清规则、通用仓库需求面积与物流库房布局。上述内容均与碳配额价格走势分析和"
           "燃煤发电企业套期保值策略建议无关。现有证据未涉及全国碳市场碳配额价格行情。")
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=True, numbers_ok=True),
                            5, 3, answer_text=ans)
    assert st == AnswerStatus.REFUSED and ref.reason == "off_topic_evidence", (st, ref)


def test_assertion_answer_not_flagged():
    """'证据表明X与Y无关'是内容断言（真答案），不得误杀。"""
    ans = "证据表明，该补偿费用与发电企业生产成本无关，由用户侧按电量分摊。"
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=True, numbers_ok=True),
                            5, 3, answer_text=ans)
    assert st == AnswerStatus.ANSWERED, ref


def test_normal_answer_not_flagged():
    ans = "发电企业月度总电费由电能量电费和辅助服务费用构成。该费用与合同容量相关。"
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=True, numbers_ok=True),
                            5, 3, answer_text=ans)
    assert st == AnswerStatus.ANSWERED, ref


def test_refusal_details_user_facing():
    """拒答话术不得暴露内部闸名（数字保真/引用校验字样）。"""
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=True, numbers_ok=False,
                                              failures=["x"]), 5, 3)
    assert st == AnswerStatus.REFUSED
    assert "数字保真" not in ref.detail
    assert "校验" not in ref.detail
    st2, ref2 = decide_status(0.9, VerifyReport(quotes_ok=False, numbers_ok=True,
                                                failures=["y"]), 5, 3)
    assert "引用校验" not in ref2.detail
    st3, ref3 = decide_status(0.3, VerifyReport(quotes_ok=True, numbers_ok=True), 5, 0)
    assert st3 == AnswerStatus.REFUSED and ref3.reason == "low_confidence"
    assert "阈值" not in ref3.detail and "宁拒勿幻" not in ref3.detail


def test_confidence_floor_when_anchored():
    """置信度误拒统一修正：硬闸全过+有已校验引用 → 自报低置信不拒（回归实锤三例）。"""
    ok = VerifyReport(quotes_ok=True, numbers_ok=True)
    # SA03#2：0.00 / S04：0.35 / SA04：0.50 —— 均不得拒
    for conf in (0.0, 0.35, 0.50):
        st, ref = decide_status(conf, ok, 5, 3)
        assert st == AnswerStatus.ANSWERED, (conf, ref)
    # 非锚定（无引用）仍按宁拒勿幻
    st, ref = decide_status(0.3, ok, 5, 0)
    assert st == AnswerStatus.REFUSED and ref.reason == "low_confidence"
    # 硬闸失败时不因置信度地板放行（先按闸拒答）
    st, ref = decide_status(0.9, VerifyReport(quotes_ok=False, numbers_ok=True), 5, 3)
    assert st == AnswerStatus.REFUSED and ref.reason == "verification_failed"
