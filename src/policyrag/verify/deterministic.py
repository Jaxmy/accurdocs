"""数字/引文确定性校验——零 LLM 成本的硬闸（计划 §三 / §五.3）。

红线：答案中每个数字/金额/日期/比率 span（含单位、方向词）必须在某
evidence 的规范化文本中逐字命中。不一致 → 删句或拒答，绝不放过。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.normalize import (
    NormalizedText,
    NumericSpan,
    canonical_date,
    canonical_number,
    extract_numeric_spans,
    normalize,
)
from ..models.answer import VerifyReport

__all__ = ["SentenceSplit", "split_sentences", "verify_numbers", "NumberFailure"]

# 引用标记（chunk_id）保护区：答案里的 [9ce4136db20f1dff] 不是数字！
# 此前 hex ID 被裸数字规则切成 "9"/"4136"/"20"/"1" 导致误杀——必须先屏蔽。
# UUID 形态 [10436a2e-7d3c-…] 与证据包装 [chunk id …] 同理（其碎片曾刷出数百条假数值失败）。
_CITE_MARK_RE = re.compile(
    r"[\[【][0-9a-fA-F]{6,64}[\]】]"                          # 带括号的 hex ID
    r"|\[[0-9a-fA-F]{6,}(?:-[0-9a-fA-F]{2,})+\]"             # UUID 形态标记
    r"|\[chunk\s+[0-9a-fA-F]+[^\]]*\]"                       # [chunk id …] 证据包装
    r"|\b(?=[0-9a-f]*[a-f])[0-9a-f]{16}\b(?=[^\w]|$)"        # 裸 16 位 hex 且含字母
)


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CITE_MARK_RE.finditer(text)]


def _is_protected(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < r_end and end > r_start for r_start, r_end in ranges)


@dataclass
class SentenceSplit:
    text: str
    start: int   # 在原文中的起点（用于删句）
    end: int


_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
# 跨句但书保护：但书句与前句合并校验（模型输出中常见"……。但书……"）
_BUT_SPLIT = re.compile(r"(但\s*书|除[此该本][^。！？]{0,30}外)")


def split_sentences(answer: str) -> list[SentenceSplit]:
    """按中文句末切分（保留但书句与前句的绑定关系——删除时整组删）。"""
    out: list[SentenceSplit] = []
    for m in _SENT_RE.finditer(answer):
        t = m.group(0).strip()
        if t:
            out.append(SentenceSplit(text=t, start=m.start(), end=m.end()))
    return out


@dataclass
class NumberFailure:
    sentence: str
    span_text: str
    canonical: str
    direction: str | None
    reason: str
    orig_start: int | None = None   # 失配 span 在答案原文中的偏移（保守删句用）
    orig_end: int | None = None
    sent_start: int | None = None   # 所属句偏移
    sent_end: int | None = None


@dataclass
class NumberCheckResult:
    ok: bool
    failures: list[NumberFailure] = field(default_factory=list)
    checked_spans: int = 0
    hit_ranges: list[tuple[int, int]] = field(default_factory=list)  # 命中 span 原文偏移


def verify_numbers(answer_text: str, evidence_texts: list[str],
                   question: str = "") -> NumberCheckResult:
    """答案中的数值 span 必须命中 evidence（规范化逐字比对 + 方向词绑定）。

    引用标记（chunk_id）不算数字，先做保护区过滤。
    question：问题文本——其数值属**用户前提**（如"2011年11月17日调研了哪家"），
    进溯源白名单，回述不作编造（R2-07）；但永不进 coverage demanded（红线 pin）。
    """
    ans_nt = normalize(answer_text)
    protected = _protected_ranges(ans_nt.norm)
    ans_spans = [
        s for s in extract_numeric_spans(ans_nt)
        if not _is_protected(s.norm_start, s.norm_end, protected)
    ]

    # evidence 侧：收集全部 canonical 数值（含方向词上下文）
    ev_canonicals: set[str] = set()
    ev_directional: set[tuple[str, str]] = set()  # (direction, canonical)
    pools = list(evidence_texts) + ([question] if question else [])
    for ev in pools:
        ev_nt = normalize(ev)
        for s in extract_numeric_spans(ev_nt):
            ev_canonicals.add(s.canonical)
            if s.direction:
                ev_directional.add((s.direction, s.canonical))
        # 日期双轨：ISO 也算命中
        for m in re.finditer(r"\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?", ev_nt.norm):
            d = canonical_date(m.group(0))
            if d:
                ev_canonicals.add(d)

    failures: list[NumberFailure] = []
    hit_ranges: list[tuple[int, int]] = []
    sentences = split_sentences(answer_text)

    for s in ans_spans:
        o_start, o_end = ans_nt.orig_span(s.norm_start, s.norm_end)
        # 溯源到所属句子（用于删句）
        sent = next(
            (x for x in sentences if x.start <= o_start < x.end),
            None,
        )
        sent_text = sent.text if sent else answer_text
        s_range = (sent.start, sent.end) if sent else (0, len(answer_text))

        if s.canonical in ev_canonicals:
            # 方向类绑定检查（计划 §五.3）：evidence 对该数值有方向主张时，
            # 答案方向类必须一致（防"上调/下调""不超过/不低于"混淆）；
            # evidence 无方向（纯陈述句）则答案补方向视为通过（宽松回退）。
            if s.direction:
                dirs_for_val = {d for (d, c) in ev_directional if c == s.canonical}
                if dirs_for_val and s.direction not in dirs_for_val:
                    failures.append(NumberFailure(
                        sentence=sent_text, span_text=s.text, canonical=s.canonical,
                        direction=s.direction,
                        reason=f"方向类『{s.direction}』与 evidence {dirs_for_val} 不符",
                        orig_start=o_start, orig_end=o_end,
                        sent_start=s_range[0], sent_end=s_range[1],
                    ))
                    continue
            hit_ranges.append((o_start, o_end))
            continue

        # 日期双轨：答案 ISO 与 evidence 原文日期比对
        d = canonical_date(s.text)
        if d and d in ev_canonicals:
            hit_ranges.append((o_start, o_end))
            continue

        failures.append(NumberFailure(
            sentence=sent_text, span_text=s.text, canonical=s.canonical,
            direction=s.direction, reason="数值未在 evidence 中逐字命中",
            orig_start=o_start, orig_end=o_end,
            sent_start=s_range[0], sent_end=s_range[1],
        ))

    return NumberCheckResult(
        ok=not failures, failures=failures, checked_spans=len(ans_spans),
        hit_ranges=hit_ranges,
    )


def strip_failing_sentences(answer: str, failures: list[NumberFailure],
                            hit_ranges: list[tuple[int, int]] | None = None) -> tuple[str, int]:
    """保守删句（R2-07）：失配句内**还有其他命中数值** → 只剥失配 span 保句；
    句内数值全失配 → 整句删。返回 (新答案, 删除句数)。

    hit_ranges 缺省时退回旧行为（整句删）——兼容旧调用。
    """
    if not failures:
        return answer, 0
    sentences = split_sentences(answer)
    bad_sents = {(f.sent_start, f.sent_end) for f in failures if f.sent_start is not None}
    if not bad_sents:                      # 旧式 failure（无偏移）→ 整句删
        bad = {f.sentence for f in failures}
        kept: list[str] = []
        removed = 0
        for s in sentences:
            if s.text in bad:
                removed += 1
            else:
                kept.append(s.text)
        return "".join(kept), removed

    hits = hit_ranges or []
    out: list[str] = []
    removed = 0
    for s in sentences:
        if (s.start, s.end) not in bad_sents:
            out.append(s.text)
            continue
        sent_hits = [h for h in hits if s.start <= h[0] < s.end]
        if not sent_hits:
            removed += 1                      # 句内数值全失配/无命中 → 整句删
            continue
        # 保句：只剥失配 span（该句内、带偏移的 failures）
        cuts = sorted(
            (f.orig_start, f.orig_end) for f in failures
            if (f.sent_start, f.sent_end) == (s.start, s.end) and f.orig_start is not None)
        merged: list[tuple[int, int]] = []
        for a, b in cuts:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        pos = s.start
        rebuilt: list[str] = []
        for a, b in merged:
            rebuilt.append(answer[pos:a])
            pos = b
        rebuilt.append(answer[pos:s.end])
        text = "".join(rebuilt).strip()
        if text:
            out.append(text)
        else:
            removed += 1
    return "".join(out), removed
