"""条款号归一化（WS-A）：第十二条 / 第12条 / 第１２条 / 12条 / Article 12 → 同一规范形。

只做**查询/词元侧**归一化——入库的 number/path/breadcrumb 原文形态与
make_chunk_id 一律不动（ID 稳定性红线）。

规范形：`{kind}:{n}`（如 `条:12`）；之N 后缀 → `条:12之2`。
英文形态（Article/Section/Clause/§）一律映射为「条」（用户语义：英文输入
对应到同一个条款）。
"""

from __future__ import annotations

import re

from ..ingest.normalize import _cn_int

__all__ = ["canon_clause_ref", "clause_ref_hits", "clause_alias_token",
           "clean_breadcrumb"]


# ---------------------------------------------------------------------------
# breadcrumb 清洗（展示/检索侧——存储与 chunk_id 一律不动）
# 数学模型附件等区域的路径段被句中碎片污染（"…第八十条 经辽宁省工业和信息化厅、 （一）"），
# 既污染结构树/引用标签，又以噪声词元进入 BM25 breadcrumb 语料（#10）。
# ---------------------------------------------------------------------------

_STRUCT_SEG_RE = re.compile(
    r"^第[0-9０-９一二三四五六七八九十百千零〇两]+[章节条款项](之[0-9０-９一二三四五六七八九十百千零〇两]+)?$"
    r"|^[（(][一二三四五六七八九十0-9]{1,3}[）)]$"
    r"|^[一二三四五六七八九十]{1,3}[、.．]$"
    r"|^\d{1,2}[、.．]\d{0,2}$"
    r"|^续\d+$"
    r"|^表$|^行\d+$"
    r"|^[fg]_\d{3}_\d{2}$"
    r"|^[A-Za-z][\.、]?$"
    r"|^附[录件]$"
)
_PUNCT_RE = re.compile(r"[、。，；：！？“”‘’（）()《》\[\]【】]")
_SENT_PUNCT_RE = re.compile(r"[。；;！!？?，,：:]")

# 标题段（结构修复后的真实路径段）：分则名/栏目名/带版本括注的长标题——
# 曾被"2-8 字干净段"规则当污染剥掉，树上分则标题整层消失
_TITLE_SEG_RE = re.compile(
    r"(规则|细则|办法|规定|通知|意见|方案|指引|准则|条例|公告|要点"
    r"|观点|主题|引言|前言|摘要|概述|综述|结论|总结|展望|总则|附则|名词解释"
    r"|行业|报告)$|（[^）]{0,24}）$|《[^》]+》"
)


def clean_breadcrumb(bc: str) -> str:
    """去掉句中碎片段，保留结构段与标题段（短干净段 / 分则名等长标题段）。"""
    kept = []
    for s in (bc or "").split(" "):
        if not s:
            continue
        if _STRUCT_SEG_RE.match(s):
            kept.append(s)
        elif 2 <= len(s) <= 8 and not _PUNCT_RE.search(s) and not s.isdigit():
            kept.append(s)   # 干净标题段（如"注册基本条件"）
        elif (len(s) <= 40 and not _SENT_PUNCT_RE.search(s)
              and not s.isdigit() and _TITLE_SEG_RE.search(s)):
            kept.append(s)   # 长标题段（分则名/栏目名/（试行…）版本括注）
    return " ".join(kept)

# 全角数字 → 半角
_FW_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")

_CN_OR_DIGIT = r"[一二三四五六七八九十百千零〇两两\d]+"

# 中文/阿拉伯混合形态：第X条(之Y)? / 第X章节款项
_PAT_CN = re.compile(
    rf"第\s*({_CN_OR_DIGIT})\s*([章节条款项])(?:\s*之\s*({_CN_OR_DIGIT}))?"
)
# 裸数字形态：12条 / 12条之2（lookbehind 防 "112条" 内部误切）
_PAT_BARE = re.compile(
    rf"(?<![\d第])({_CN_OR_DIGIT})\s*([章节条款项])(?:\s*之\s*({_CN_OR_DIGIT}))?"
)
# 英文形态：Article 12 / Art. 12 / Section 12 / Clause 12 / §12（映射为「条」）
_PAT_EN = re.compile(
    r"(?:Article|Art\.?|Section|Sec\.?|Clause|§)\s*\.?\s*(\d{1,4})(?:\s*之\s*(\d{1,3}))?",
    re.IGNORECASE,
)

# 用于从整段文本中"找出全部条款形 span"的组合扫描
_SPAN_SCAN_RE = re.compile(
    rf"第\s*{_CN_OR_DIGIT}\s*[章节条款项](?:\s*之\s*{_CN_OR_DIGIT})?"
    rf"|(?<![\d第]){_CN_OR_DIGIT}\s*[章节条款项](?:\s*之\s*{_CN_OR_DIGIT})?"
    rf"|(?:Article|Art\.?|Section|Sec\.?|Clause|§)\s*\.?\s*\d{{1,4}}(?:\s*之\s*\d{{1,3}})?",
    re.IGNORECASE,
)


def _num(s: str | None) -> int | None:
    if not s:
        return None
    s = s.translate(_FW_TRANS).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return _cn_int(s)
        except Exception:  # noqa: BLE001
            return None


def canon_clause_ref(text: str) -> str | None:
    """条款引用 → 规范形（`条:12` / `条:12之2` / `章:3` …）。非条款引用返回 None。"""
    if not text:
        return None
    t = text.translate(_FW_TRANS).strip()

    m = _PAT_CN.search(t)
    if m:
        n, kind, sub = _num(m.group(1)), m.group(2), _num(m.group(3))
        if n is not None:
            return f"{kind}:{n}" + (f"之{sub}" if sub else "")
    m = _PAT_EN.search(t)
    if m:
        n, sub = _num(m.group(1)), _num(m.group(2))
        if n is not None:
            return f"条:{n}" + (f"之{sub}" if sub else "")
    m = _PAT_BARE.search(t)
    if m:
        n, kind, sub = _num(m.group(1)), m.group(2), _num(m.group(3))
        if n is not None:
            return f"{kind}:{n}" + (f"之{sub}" if sub else "")
    return None


def clause_ref_hits(ref: str, haystack: str) -> bool:
    """边界感知条款匹配：规范形相等才算命中（`条:12` 不命中"112条"）。

    ref 不是条款形 → 退回子串匹配（非条款 anchor 语义不变）。
    """
    canon = canon_clause_ref(ref)
    if canon is None:
        return ref in (haystack or "")
    hay = (haystack or "").translate(_FW_TRANS)
    for m in _SPAN_SCAN_RE.finditer(hay):
        if canon_clause_ref(m.group(0)) == canon:
            return True
    return False


def clause_alias_token(text: str) -> str | None:
    """条款形 span → BM25 别名词元（`第十二条` → `§条12`，双侧对称发射）。"""
    canon = canon_clause_ref(text)
    return ("§" + canon.replace(":", "")) if canon else None
