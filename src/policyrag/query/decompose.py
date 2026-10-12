"""多问题拆分：批量提问（10-20 问）→ 独立子问题列表。

引擎对每个子问题走完整单题链路（检索/推理/校验），互不挤占 top-8 证据位。
拆分确定性优先（零 LLM 成本、可回放）：

1. 编号列表（1. / 2、 / （3） / 4) …）→ 按编号切，导语拼到每个子问题前；
2. 多行问句（每行以？/？结尾）→ 按行切；
3. 行内多问号（A是多少？B是多少？）→ 按问号边界切；
4. 都不命中 → 原文即单问题。

指代开头的子问题（它/其/该…）自动附前一问作上下文（部分相关问题）。
"""

from __future__ import annotations

import re

__all__ = ["decompose_questions"]

# 行首编号：1. 2、 3． （4） 5) 一、 第一， ① ② …
_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"第?[0-9０-９一二三四五六七八九十百]{1,3}[、.．:：)）,，]|"
    r"[（(][0-9０-９一二三四五六七八九十百]{1,3}[)）]|"
    r"[0-9０-９]{1,2}[.．、)）]|"
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]|"
    r"[一二三四五六七八九十]{1,3}[、.．,，]"
    r")\s*"
)

_ANAPHORA_RE = re.compile(
    r"^(它|它们|其|该|这个|这些|那个|那些|此|上述|上文|前面|同上"
    r"|如何|怎么|怎样|为何|为什么|何以|是不是|是否|能否|可否|是多少|为多少|多少|几(?!乎|何))"
)

# 无主语续问（"如何计算？"）：语义依附前问，不许独立成题
_SUBJECTLESS_RE = re.compile(
    r"^(如何|怎么|怎样|为何|为什么|何以|是不是|是否|能否|可不可以|可否"
    r"|是多少|为多少|多少|几[个种档类段倍年月家次点]|几(?!乎|何))"
)

_QUESTION_MARK_RE = re.compile(r"[？?]")

MAX_QUESTIONS = 24


def decompose_questions(text: str) -> list[str]:
    """批量提问 → 子问题列表。单问题返回 [原文]；空输入返回 []。"""
    t = (text or "").strip()
    if not t:
        return []
    for splitter in (_split_numbered, _split_by_lines, _split_inline_marks):
        qs = splitter(t)
        if qs is not None and len(qs) >= 2:
            # 编号列表尊重用户显式拆分（只补指代上下文）；启发式切分合并无主语续问
            return _finalize(qs, limit=MAX_QUESTIONS,
                             merge_cont=(splitter is not _split_numbered))
    return [t]


def _merge_continuations(parts: list[str]) -> list[str]:
    """无主语续问并入前问："X是怎么规定的？如何计算？" 是一个复合问题。

    否则"如何计算？"独立检索会漂移到任意公式块（实测 bug）。
    """
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and _SUBJECTLESS_RE.match(p) and len(p) <= 20:
            out[-1] = f"{out[-1]}{p}"
        else:
            out.append(p)
    return out


def _finalize(qs: list[str], limit: int, merge_cont: bool = True) -> list[str]:
    """去空、无主语续问并题（启发式切分时）、指代补上下文、超限合并尾部。"""
    if merge_cont:
        qs = _merge_continuations(qs)
    out: list[str] = []
    prev = ""
    for q in qs:
        q = re.sub(r"\s+", " ", q).strip(" \t\r\n·")
        if not q:
            continue
        if prev and _ANAPHORA_RE.match(q):
            q = f"（相关前问：{prev}）{q}"
        out.append(q)
        prev = q
    if len(out) > limit:
        head, tail = out[: limit - 1], "；".join(out[limit - 1:])
        out = head + [tail]
    return out or []


def _split_numbered(t: str) -> list[str] | None:
    lines = t.splitlines()
    items: list[list[str]] = []
    preamble: list[str] = []
    for ln in lines:
        m = _NUM_PREFIX_RE.match(ln)
        if m and len(ln[m.end():].strip()) >= 2:
            items.append([ln[m.end():].strip()])
        elif items:
            items[-1].append(ln.strip())   # 续行归属当前问题
        else:
            preamble.append(ln.strip())
    if len(items) < 2:
        return None
    qs = [" ".join(parts) for parts in items]
    pre = " ".join(p for p in preamble if p)
    if pre:
        qs = [f"{pre} {q}" for q in qs]
    return qs


def _split_by_lines(t: str) -> list[str] | None:
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    q_lines = [ln for ln in lines if _QUESTION_MARK_RE.search(ln)]
    if len(q_lines) >= 2 and len(q_lines) == len(lines):
        return q_lines
    return None


def _split_inline_marks(t: str) -> list[str] | None:
    if len(_QUESTION_MARK_RE.findall(t)) < 2:
        return None
    flat = re.sub(r"\s+", "", t)
    parts: list[str] = []
    last = 0
    for m in _QUESTION_MARK_RE.finditer(flat):
        seg = flat[last:m.end()]
        if seg:
            parts.append(seg)
        last = m.end()
    tail = flat[last:]
    if tail:
        # 尾随片段：像问句（含疑问词）独立成题，否则并入上一问
        if re.search(r"(是多少|是什么|哪些|如何|怎么|几[个项类倍年月]|是否|能否|多少)", tail):
            parts.append(tail)
        elif parts:
            parts[-1] += tail
    parts = [p for p in parts if len(p) >= 4] or parts
    return parts if len(parts) >= 2 else None
