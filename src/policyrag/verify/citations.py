"""引用校验：ID 存在性、quote 逐字命中、引用完备性（计划 §三 verify/citations）。

阶梯（R2-10）：逐字折叠包含 → LaTeX 剥离 → 非词剥离 → 词元包含 →
快照吸附（对齐到真源句窗，保逐字）→ 失败。降级转述（剥引号声明）由
pipeline 在"全答数值可溯源"前提下最后手段执行，此处不放水。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ..ingest.normalize import (
    canonical_number,
    extract_numeric_spans,
    map_symbol_pua,
    normalize,
)
from ..models.answer import Citation, VerifyReport

__all__ = ["verify_citations", "CitationFailure", "strip_unknown_cite_marks",
           "sanitize_citations", "demote_unmatched_quotes"]


@dataclass
class CitationFailure:
    chunk_id: str
    reason: str
    quote: str = ""


@dataclass
class CitationCheckResult:
    ok: bool
    failures: list[CitationFailure] = field(default_factory=list)
    verified_count: int = 0
    snapped_count: int = 0   # 快照吸附修复的引文数（对齐到真源）


# 非法引用标记形态：编造 UUID、[chunk …] 证据包装、不在白名单的 hex
# （模型曾编造这些形态致整答被拒，其碎片还污染数值闸——校验前一律净化）
_UUID_MARK_RE = re.compile(r"\s*\[[0-9a-fA-F]{6,}(?:-[0-9a-fA-F]{2,})+\]")
_CHUNK_WRAP_RE = re.compile(r"\s*\[chunk\s+[0-9a-fA-F]+[^\]]*\]")
_HEX_MARK_RE = re.compile(r"\s*\[([0-9a-fA-F]{6,64})\]")


def strip_unknown_cite_marks(text: str, known_ids: set[str]) -> str:
    """剥离一切非白名单引用标记（UUID/[chunk 包装]/假 hex）——保留正文。"""
    if not text:
        return text
    t = _UUID_MARK_RE.sub("", text)
    t = _CHUNK_WRAP_RE.sub("", t)

    def _hex(m: re.Match) -> str:
        return "" if m.group(1) not in known_ids else m.group(0)

    return _HEX_MARK_RE.sub(_hex, t)


def sanitize_citations(citations: list[Citation], registry: dict[str, str]) -> list[Citation]:
    """剔除 registry 之外的伪造引用条目（假 id 不得触发整答拒答）。"""
    return [c for c in citations if c.chunk_id in registry]


def demote_unmatched_quotes(citations: list[Citation],
                            failed: list[CitationFailure],
                            registry: dict[str, str] | None = None) -> list[CitationFailure]:
    """F4c 降级转述（最后手段，调用方前提：全答数值已过 verify_numbers）：
    失配引文剥掉引文声明（quote 清空、标记 demoted）——答案不再声称逐字。

    红线守卫：引文自身的数值必须全部落在其**数值作用域**（registry 提供；
    pipeline 侧把同表行兄弟文本并入该块作用域——表按行切块但表是一整个语义单元）。
    引文携带无出处数值的**不降级**，留作硬失败拒答（test_blocks_fabricated_quote）。
    返回剩余硬失败列表。
    """
    hard: list[CitationFailure] = []
    by_id = {c.chunk_id: c for c in citations}
    for f in failed:
        if "疑似引用幻觉" not in f.reason:
            hard.append(f)
            continue
        c = by_id.get(f.chunk_id)
        if c is None or c.verified:
            hard.append(f)
            continue
        if registry is not None:
            q_canons = {s.canonical for s in extract_numeric_spans(normalize(f.quote or c.quote))}
            w_canons = {s.canonical for s in extract_numeric_spans(
                normalize(registry.get(f.chunk_id, "")))}
            if not q_canons.issubset(w_canons):
                hard.append(f)      # 引文数值无出处 → 不降级，硬拒
                continue
        c.quote = ""
        c.demoted = True
    return hard


def _tokens(s: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z一-鿿]+", s)


_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def _snap_quote(quote: str, chunk_text: str) -> str | None:
    """F4b 快照吸附：quote 对齐到 chunk 内最佳匹配句窗（对齐到真源，保逐字）。

    守卫（缺一不吸）：①长度归一相似度 ≥0.85；②唯一最佳（领先次优 ≥0.05）；
    ③句窗含 quote 的全部 canonical 数值（数值同一性——防把引文"改写"成另一句）。
    """
    if not quote or not chunk_text:
        return None
    q_norm = re.sub(r"\s+", "", map_symbol_pua(normalize(quote).norm))
    if not q_norm:
        return None
    q_canons = {s.canonical for s in extract_numeric_spans(normalize(quote))}

    sents = [m.group(0).strip() for m in _SENT_RE.finditer(chunk_text)]
    sents = [s for s in sents if s]
    windows: list[str] = []
    for i, s in enumerate(sents):
        windows.append(s)
        if i + 1 < len(sents):
            windows.append(s + sents[i + 1])   # 跨句引文窗

    scored: list[tuple[float, str]] = []
    for w in windows:
        w_norm = re.sub(r"\s+", "", map_symbol_pua(normalize(w).norm))
        if not w_norm:
            continue
        ratio = difflib.SequenceMatcher(None, q_norm, w_norm).ratio()
        scored.append((ratio, w))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_r, best_w = scored[0]
    second_r = scored[1][0] if len(scored) > 1 else 0.0
    if best_r < 0.85 or (best_r - second_r) < 0.05:
        return None
    # 数值同一性：吸附窗必须含 quote 全部 canonical 数值
    w_canons = {s.canonical for s in extract_numeric_spans(normalize(best_w))}
    if not q_canons.issubset(w_canons):
        return None
    return best_w.strip()


def verify_citations(
    citations: list[Citation],
    registry: dict[str, str],
    retrieved: set[str],
    answer_text: str = "",
) -> CitationCheckResult:
    """多层校验：
    ① chunk_id 在 registry 中存在；
    ② chunk_id ∈ 本次检索/展开集合（防模型编造"库里有但没检索到"的引用）；
    ③ quote 归一化后是 chunk 归一化文本的子串（逐字命中，防引用幻觉）；
    ④ 宽松层（全 kind，R2-10 修）：LaTeX 装饰剥离 → 符号剥离 → 词元包含度（≥0.6）；
    ⑤ 快照吸附：相似度+唯一最佳+数值同一性守卫下对齐到原文句窗（保逐字）。
    """
    failures: list[CitationFailure] = []
    verified = 0
    snapped = 0

    for c in citations:
        if c.chunk_id not in registry:
            failures.append(CitationFailure(c.chunk_id, "chunk_id 不存在于 registry", c.quote))
            continue
        if retrieved and c.chunk_id not in retrieved:
            failures.append(CitationFailure(
                c.chunk_id, "chunk_id 不在本次检索/展开集合中", c.quote
            ))
            continue

        # 逐字命中比对：空白全部折叠后再子串匹配（防"8%"vs"8 %"排版差异误判）
        chunk_text = registry[c.chunk_id]
        chunk_norm = re.sub(r"\s+", "", map_symbol_pua(normalize(chunk_text).norm))
        quote_norm = re.sub(r"\s+", "", map_symbol_pua(normalize(c.quote).norm))
        if quote_norm and quote_norm not in chunk_norm:
            # ⑤ 快照吸附（优先修复）：对齐到真源句窗——quote 改写成原文，
            # 显示即逐字（守卫：相似度/唯一最佳/数值同一性，缺一不吸）
            snapped_span = _snap_quote(c.quote, chunk_text)
            if snapped_span is not None:
                c.quote = snapped_span
                snapped += 1
                c.verified = True
                verified += 1
                continue
            # 宽松层：剥 LaTeX 装饰后匹配（公式块）；表格断词/长文本块同享
            chunk_loose = re.sub(r"\\left|\\right|\\mathrm|[{}\\\s]", "", chunk_norm)
            quote_loose = re.sub(r"\\left|\\right|\\mathrm|[{}\\\s]", "", quote_norm)
            if not (quote_loose and quote_loose in chunk_loose):
                # 符号宽松层：Symbol 字体乱码公式（PUA≥× 等）被模型引用时
                # 常被丢弃不可见字符——两侧剥符号后仍须子串命中（字词数字保留）
                chunk_alnum = re.sub(r"[\W_]+", "", chunk_norm, flags=re.UNICODE)
                quote_alnum = re.sub(r"[\W_]+", "", quote_norm, flags=re.UNICODE)
                if not (quote_alnum and quote_alnum in chunk_alnum):
                    # 词元包含层（重排/转写形态差）：quote 词元 ≥60% 落在 chunk 中
                    q_toks = _tokens(quote_norm)
                    hit = sum(1 for t in q_toks if t in chunk_norm)
                    if not (q_toks and hit / len(q_toks) >= 0.6):
                        failures.append(CitationFailure(
                            c.chunk_id, "quote 未逐字命中 chunk 原文（疑似引用幻觉）", c.quote
                        ))
                        continue

        c.verified = True
        verified += 1

    return CitationCheckResult(ok=not failures, failures=failures,
                               verified_count=verified, snapped_count=snapped)