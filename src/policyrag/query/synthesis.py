"""两阶段编排的综合推理引擎（WS-C）：基于已验证答案的文档锚定推导。

形态（用户决策：文档锚定推导）：
- 流式正文 + 行内标记（G6）：事实句"（据问题 N）"，推导句"【推导】…[basis:1,2]"；
- verify_synthesis（G4）：字面数值须命中允许集；推导数值须可由允许集数值
  确定性闭包复算（加/减/百分点差/比值换百分比）且带【推导】标；否则删句；
- 禁止新增原子编码引用；超文档知识须显式标注段且不含具体数字/条款。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.normalize import canonical_number, normalize, extract_numeric_spans
from ..llm.base import ChatMessage

__all__ = ["run_synthesis", "verify_synthesis", "SynthesisReport", "parse_synthesis_text"]

SYSTEM_SYNTHESIS = """你是分析推理引擎（不是检索引擎）。基于【已验证事实】回答用户的分析推理要求。

红线：
1. 只能从【已验证事实】出发推导。事实句句末标注来源"（据问题 N）"；推导句以【推导】开头，句末用 [basis:题号,题号] 标注依据。
2. 数值只允许：(a) 从【已验证事实】原文复制；(b) 由其中数值经加/减/百分点差/比值换算得出——(b) 类数值必须带【推导】标注。
3. 禁止引入新的原子编码（十六进制 ID）引用；引用事实写"（据问题 N）"即可。
4. 需要文档之外领域知识时，单独成段，以"（文档未覆盖，以下为一般性分析）"开头；该段不得出现具体文档数字或条款号。
5. 事实不足以支撑推导时明说"基于现有事实无法推导"，不要硬编。

直接输出正文（不要 JSON、不要 markdown 代码块）。"""

_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_BASIS_RE = re.compile(r"\[basis[:：]\s*([0-9,\s，]+)\]")
_DERIVE_MARK = "【推导】"
_CITE_ID_RE = re.compile(r"\[[0-9a-fA-F]{16}\]")


@dataclass
class SynthesisReport:
    ok: bool
    dropped: list[str] = field(default_factory=list)
    inference_ok: int = 0
    literal_ok: int = 0
    basis_qids: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@dataclass
class _Values:
    """允许集：字面数值 canonical（严格逐字）+ 供闭包复算的 float 集。"""
    texts: list[str]
    floats: set[float]
    canon: set[str]

    def allows_literal(self, canonical: str) -> bool:
        """字面值判定只认 canonical 命中（跨单位 float 匹配会误放行）。"""
        return canonical in self.canon


def _to_float(canonical: str) -> float | None:
    s = canonical.replace(",", "").replace("，", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _collect_values(texts: list[str]) -> _Values:
    floats: set[float] = set()
    canon: set[str] = set()
    for t in texts:
        nt = normalize(t)
        for s in extract_numeric_spans(nt):
            canon.add(s.canonical)
            v = _to_float(s.canonical)
            if v is not None:
                floats.add(v)
    return _Values(texts=list(texts), floats=floats, canon=canon)


def _closure_reproduces(value: float, allowed: _Values) -> bool:
    """推导值可由允许集两两闭包复算（加/减/百分点差/比值换百分比）？"""
    vals = sorted(allowed.floats)
    if value in vals:
        return True
    for a in vals:
        for b in vals:
            cands = (a + b, a - b, b - a, a * b,
                     (a / b if b else None), (b / a if a else None),
                     (a / b * 100 if b else None), (b / a * 100 if a else None))
            for c in cands:
                if c is not None and abs(c - value) <= 1e-6 * max(1.0, abs(value)):
                    return True
    return False


def parse_synthesis_text(text: str) -> dict:
    """流式正文 → 结构（确定性解析行内标记）。"""
    # 标记归位："[basis:…]" 若落在句末标点后，移回句内（绑定前句）
    text = re.sub(r"([。！？!?；;])\s*(\[basis[:：][^\]]*\])", r"\2\1", text)
    basis: list[str] = []
    inferences: list[dict] = []
    factual: list[str] = []
    general_note = False
    for sent in _SENT_RE.findall(text):
        s = sent.strip()
        if not s:
            continue
        if "文档未覆盖" in s:
            general_note = True
        refs = []
        for m in _BASIS_RE.finditer(s):
            refs += [x.strip() for x in re.split(r"[,，]", m.group(1)) if x.strip()]
        clean = _BASIS_RE.sub("", s).strip()
        if _DERIVE_MARK in clean:
            inferences.append({
                "text": clean.replace(_DERIVE_MARK, "").strip(),
                "based_on": refs, "confidence": 0.6 if refs else 0.4,
            })
        else:
            factual.append(clean)
        basis += refs
    return {
        "analysis": text.strip(),
        "basis_questions": sorted(set(basis), key=lambda x: (len(x), x)),
        "factual_claims": factual,
        "inferences": inferences,
        "uncertainties": [] if not general_note else ["含文档外一般性分析段"],
        "has_general_note": general_note,
    }


def verify_synthesis(text: str, basis_texts: list[str],
                     basis_chunk_ids: set[str] | None = None) -> tuple[str, SynthesisReport]:
    """合成硬闸：字面数值溯源 + 推导闭包复算 + 引用禁令。返回 (清理后文本, 报告)。"""
    allowed = _collect_values(basis_texts)
    report = SynthesisReport(ok=True,
                             basis_qids=parse_synthesis_text(text)["basis_questions"])
    basis_chunk_ids = basis_chunk_ids or set()
    # 标记归位（与 parse 同步）：纯标记残片不许单独成句存活
    text = re.sub(r"([。！？!?；;])\s*(\[basis[:：][^\]]*\])", r"\2\1", text)

    kept: list[str] = []
    for sent in _SENT_RE.findall(text):
        s = sent.strip()
        if not s:
            continue
        # 引用禁令：不在 basis 已引集合的原子编码 → 剥离
        for m in _CITE_ID_RE.finditer(s):
            if m.group(0)[1:-1] not in basis_chunk_ids:
                report.failures.append(f"禁止新增原子编码引用: {m.group(0)}")
        s_clean = _CITE_ID_RE.sub("", s)

        is_derived = _DERIVE_MARK in s_clean
        # 来源标注（据问题 N）里的序号不是数值——提取 span 前剥离
        nt = normalize(re.sub(r"（据问题[^）]*）|\[basis[:：][^\]]*\]|【推导】",
                              "", s_clean))
        spans = extract_numeric_spans(nt)
        ok = True
        for sp in spans:
            v = _to_float(sp.canonical)
            if allowed.allows_literal(sp.canonical):
                report.literal_ok += 1
                continue
            if is_derived and v is not None and _closure_reproduces(v, allowed):
                report.inference_ok += 1
                continue
            ok = False
            report.failures.append(
                f"数值不可溯源{'（推导不可复算或未标【推导】）' if is_derived else ''}: {sp.text}")
            break
        if ok:
            kept.append(s_clean)
        else:
            report.dropped.append(s_clean)
            report.ok = False
    return "".join(kept), report


def run_synthesis(llm, question: str, basis_facts: list[dict],
                  on_delta=None, max_tokens: int = 2048) -> dict:
    """综合推理主入口。basis_facts: [{qid, question, answer, evidence: [原文...]}]。

    零 basis → 直接拒答（G5 红线：不给模型自由发挥）。
    """
    if not basis_facts:
        return {
            "status": "rejected",
            "text": "",
            "basis_questions": [],
            "inferences": [],
            "factual_claims": [],
            "uncertainties": [],
            "verification": {"ok": False,
                             "failures": ["无已验证事实基础，拒绝推导"]},
        }

    facts_block = "\n\n".join(
        f"【问题 {f['qid']}】{f['question']}\n【已验证答案】{f['answer']}\n"
        f"【原文摘录】\n" + "\n".join(f"- {e}" for e in (f.get("evidence") or [])[:6])
        for f in basis_facts
    )
    messages = [
        ChatMessage(role="system", content=SYSTEM_SYNTHESIS, cache=True),
        ChatMessage(role="user",
                    content=f"【已验证事实】\n{facts_block}\n\n"
                            f"【分析推理要求】\n{question}"),
    ]
    result = llm.complete(messages, max_tokens=max_tokens, effort="high",
                          on_delta=on_delta)
    text = result.text or ""
    parsed = parse_synthesis_text(text)
    cleaned, report = verify_synthesis(
        text, [f["answer"] for f in basis_facts]
        + [e for f in basis_facts for e in (f.get("evidence") or [])],
    )
    status = "ok" if (report.ok or cleaned.strip()) else "rejected"
    if report.dropped and cleaned.strip():
        status = "partial"
    return {
        "status": status,
        "text": cleaned,
        "analysis_raw": text,
        "basis_questions": parsed["basis_questions"],
        "factual_claims": parsed["factual_claims"],
        "inferences": parsed["inferences"],
        "uncertainties": parsed["uncertainties"],
        "verification": {"ok": report.ok, "failures": report.failures,
                         "dropped": report.dropped[:5],
                         "literal_ok": report.literal_ok,
                         "inference_ok": report.inference_ok},
        "usage": {"output_tokens": getattr(result.usage, "output_tokens", 0)},
    }
