"""展示分层（WS-G）：原子 ID 对用户隐藏。

引擎内部绑定不动（[chunk_id] 行内标记支撑 B7 硬闸）——本模块只在
adapter 边界剥离编码、拆 display/ref 两层、生成句↔引用对位（line_refs）。

原则：只有引擎收到复核验证要求时才使用原子编码（policy_verify /
require_review / expose_ids 显式取回）。
"""

from __future__ import annotations

import re

__all__ = ["present_for_user", "strip_id_marks"]

_MARK_RE = re.compile(r"\s*\[([0-9a-fA-F]{6,64})\]")
# 一切证据包装/伪造标记形态：UUID、[chunk …] 包装——用户可见文本一律剥离
_WRAP_MARK_RE = re.compile(
    r"\s*\[[0-9a-fA-F]{6,}(?:-[0-9a-fA-F]{2,})+\]"
    r"|\s*\[chunk\s+[0-9a-fA-F]+[^\]]*\]")
_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def strip_id_marks(text: str) -> str:
    """剥离原子编码/证据包装标记并清理标点空隙。"""
    out = _WRAP_MARK_RE.sub("", text or "")
    out = _MARK_RE.sub("", out)
    out = re.sub(r"\s+([，。、；：）】》！？])", r"\1", out)
    out = re.sub(r"([（【《])\s+", r"\1", out)
    return out


def _present_answer(answer: str, id_to_idx: dict[str, int]) -> tuple[str, list[dict]]:
    """answer → (去编码正文, line_refs)。line_refs[i] = 句序 → 引用下标列表。"""
    # 标记归位："[id]" 若落在句末标点后，移回句内（绑定前句）
    answer = re.sub(r"([。！？!?；;])\s*(\[[0-9a-fA-F]{6,64}\])", r"\2\1", answer)
    display_parts: list[str] = []
    line_refs: list[dict] = []
    for sent in _SENT_RE.findall(answer or ""):
        ids = _MARK_RE.findall(sent)
        clean = strip_id_marks(sent)
        refs = []
        for cid in ids:
            if cid in id_to_idx:
                refs.append(id_to_idx[cid])
        if clean.strip():
            display_parts.append(clean)
            if refs:
                line_refs.append({"sentence_no": len(display_parts) - 1,
                                  "cite_indexes": sorted(set(refs))})
        elif refs and line_refs:
            # 纯标记尾缀：并入上一句的引用
            line_refs[-1]["cite_indexes"] = sorted(
                set(line_refs[-1]["cite_indexes"] + refs))
    return "".join(display_parts), line_refs


def present_for_user(out: dict, expose_ids: bool = False) -> dict:
    """envelope dict → 用户视图（answer 去编码 + citations display/ref 分层）。"""
    if expose_ids:
        return out
    out = dict(out)
    cites = out.get("citations") or []
    id_to_idx: dict[str, int] = {}
    for i, c in enumerate(cites):
        id_to_idx.setdefault(str(c.get("chunk_id") or ""), i)

    answer, line_refs = _present_answer(out.get("answer") or "", id_to_idx)
    out["answer"] = answer
    out["line_refs"] = line_refs

    new_cites = []
    for c in cites:
        cc = dict(c)
        cid = str(cc.pop("chunk_id", "") or "")
        # #10 展示侧：引用标签/路径去句中碎片段
        if cc.get("clause_path"):
            from ..models.clause_norm import clean_breadcrumb
            cc["clause_path"] = clean_breadcrumb(str(cc["clause_path"]))
        new_cites.append({"display": cc, "ref": {"chunk_id": cid}})
    out["citations"] = new_cites

    # 逐题结果递归去码（meta.per_question 不泄漏）
    meta = dict(out.get("meta") or {})
    if meta.get("per_question"):
        meta["per_question"] = [present_for_user(p) for p in meta["per_question"]]
    out["meta"] = meta

    # synthesis 文本本应无码（硬闸剥离过）——防御性再剥一次
    if out.get("synthesis"):
        syn = dict(out["synthesis"])
        syn["text"] = strip_id_marks(syn.get("text") or "")
        out["synthesis"] = syn
    return out
