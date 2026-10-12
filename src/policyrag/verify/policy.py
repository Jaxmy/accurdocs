"""拒答决策与 risk-tier 路由（计划 §四 B1/B5、§五.17）。

拒答 precision/recall 双指标：应拒拒了 / 不该拒没拒。阈值由 eval 调。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.answer import AnswerStatus, Refusal, VerifyReport

__all__ = ["RiskTier", "route_risk_tier", "decide_status", "CONFIDENCE_THRESHOLD"]

CONFIDENCE_THRESHOLD = 0.55   # MVP 默认；eval 后调

# 简单事实题特征：单实体 + 明确条款号/单一查询意图
# 条款形态含裸数字条（"12条"）与全角数字（"第１２条"）——WS-A 归一化配套
_CLAUSE_REF_RE = re.compile(
    r"(?:第[一二三四五六七八九十百千零〇\d０-９]+条(?:之[一二三四五六七八九十\d]+)?"
    r"|(?<![\d第])\d{1,3}条)"
)
_NUMERIC_Q_RE = re.compile(
    r"(多少|几个|几条|金额|比例|费率|阈值|上限|下限|期限|日期|何时"
    r"|怎么算|怎么计算|如何算|如何计算|计算方式|计算公式|公式怎么|怎么计)"
)


class RiskTier:
    SHORT = "short"   # 简单事实：一次调用、可跳 rerank/审查
    FULL = "full"     # 综合分析：全链路


def route_risk_tier(query: str, mode: str) -> str:
    """B1 风险分流。quote 模式 + 单条款引用 + 无数值综合 → SHORT。"""
    if mode == "analyze":
        return RiskTier.FULL
    has_clause = bool(_CLAUSE_REF_RE.search(query))
    simple_numeric = bool(_NUMERIC_Q_RE.search(query)) and len(query) < 60
    # 单一明确意图：短问句 + 条款号/数值锚点
    if len(query) < 80 and (has_clause or simple_numeric):
        return RiskTier.SHORT
    return RiskTier.FULL


# "证据无关"式声明：正文只是说'证据与问题无关/未涉及'——不是答案（#7 拒答语义）。
# 判定 = 开头主语（证据/原文…）+ 声明动词（仅涉及/未涉及…，多句式任意位置）——
# "证据表明X与Y无关"是内容断言（真答案），无声明动词不误杀。
_DISCLAIMER_SUBJ = re.compile(r"^\s*(?:现有)?(?:证据|原文|文档|材料|库内|检索)")
_DISCLAIMER_PRED = re.compile(
    r"(?:仅|只)(?:涉及|提及|包括|包含)"          # "证据仅涉及A、B、C"（枚举无关项）
    r"|(?:未|没有|不)(?:涉及|覆盖|包含)"          # "未涉及碳价走势"
    r"|(?:均|都)与[^。；]{0,40}(?:无关|不相关)"    # "均与…无关"（裸"与…无关"是内容断言不采）
    r"|无法回答|不含")


def decide_status(
    confidence: float,
    verification: VerifyReport,
    evidence_count: int,
    citations_count: int,
    answer_text: str = "",
) -> tuple[AnswerStatus, Refusal | None]:
    """B5/B7/B8 汇总后的最终状态决策。"""
    # 声明式非答案："现有证据仅涉及…均与问题无关"→ 必须拒答（曾被误判 answered）；
    # 多句式长声明（>220 字）也曾绕过——长度上限放宽到 500
    ans = (answer_text or "").strip()
    if ans and len(ans) < 500 and _DISCLAIMER_SUBJ.match(ans) and _DISCLAIMER_PRED.search(ans):
        return AnswerStatus.REFUSED, Refusal(
            reason="off_topic_evidence",
            detail="已检索到的内容与所问主题无关，知识库未覆盖该问题。",
        )
    if evidence_count == 0:
        return AnswerStatus.REFUSED, Refusal(
            reason="insufficient_evidence",
            detail="检索未命中任何相关条款，无法在文档范围内回答。",
        )
    if not verification.quotes_ok:
        return AnswerStatus.REFUSED, Refusal(
            reason="verification_failed",
            detail="为保准确：答案中的引文无法逐字回溯到原文，已拒答。",
            missing_hints=verification.failures[:5],
        )
    if not verification.numbers_ok:
        return AnswerStatus.REFUSED, Refusal(
            reason="verification_failed",
            detail="为保准确：答案中的数值无法逐字回溯到原文，已拒答。",
            missing_hints=verification.failures[:5],
        )
    if verification.review_ok is False:
        return AnswerStatus.REFUSED, Refusal(
            reason="adversarial_review_failed",
            detail="对抗审查判定存在无原文支撑的断言。",
            missing_hints=verification.failures[:5],
        )
    # 置信度误拒统一修正（双入口回归实锤：SA03#2=0.00/S04=0.35/SA04=0.50 均硬闸全过被拒）：
    # 走到此处说明引文逐字回溯、数值保真、覆盖、审查全过——答案已被原文锚定，
    # 模型自报置信度（OCR/公式证据常见 0.0-0.5 主观犹豫）不作拒答依据。
    # 非锚定态（无已校验引用）保留低置信拒答（宁拒勿幻）。
    anchored = (verification.quotes_ok and verification.numbers_ok
                and verification.review_ok is not False and citations_count > 0)
    if confidence < CONFIDENCE_THRESHOLD and not anchored:
        return AnswerStatus.REFUSED, Refusal(
            reason="low_confidence",
            detail=(f"证据把握不足（置信度 {confidence:.2f}），"
                    "按宁可拒答、不可幻觉原则拒答。"),
        )
    if citations_count == 0:
        return AnswerStatus.REFUSED, Refusal(
            reason="no_citation",
            detail="答案缺少可校验的原文引用。",
        )
    return AnswerStatus.ANSWERED, None
