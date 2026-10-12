"""中文检索分词：jieba + 条款号/文号/金额/日期正则预切。

政策/金融检索的精确匹配靠这里：`〔2023〕1号`、`第三条`、`1,000 万元`
必须成为完整 token，不能被 jieba 切碎。
"""

from __future__ import annotations

import re

import jieba

__all__ = ["tokenize", "precut", "TOKENIZER_VERSION"]

# 分词规则版本：预切/别名词元逻辑变更必须 bump——BM25 corpus.json 据此失效重建
TOKENIZER_VERSION = "3"

# 预切保护 token（先于 jieba）；条款号支持全角数字/裸数字/英文形态（WS-A）
_PRECUT_RE = re.compile(
    r"〔\d{2,4}〕\d+号?"                                  # 文号
    r"|第[一二三四五六七八九十百千零〇\d０-９]+条(?:之[一二三四五六七八九十\d０-９]+)?"  # 条款号
    r"|第[一二三四五六七八九十百千零〇\d０-９]+[章节款项]"          # 章节款
    r"|(?<![\d第０-９])\d{1,4}[章节条款项](?:之\d{1,3})?"   # 裸数字条（12条；lookbehind 防"112条"误切）
    r"|(?:Article|Art\.?|Section|Sec\.?|Clause|§)\s*\.?\s*\d{1,4}(?:\s*之\s*\d{1,3})?"  # 英文条款
    r"|\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?"                # 日期
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:万元|亿元|元|万|亿)?"  # 千分位金额
    r"|\d+(?:\.\d+)?(?:万元|亿元|元|%|％|个百分点|bp|基点)"   # 数值+单位
    r"|\d+(?:\.\d+)?"                                    # 裸数字
)


def precut(text: str) -> list[str]:
    """把保护 span 先切出来，剩余文本留给 jieba。"""
    out: list[str] = []
    pos = 0
    for m in _PRECUT_RE.finditer(text):
        if m.start() > pos:
            out.append(text[pos : m.start()])
        out.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_SYMNAME_RE = re.compile(r"_[{\[]([^}\]]{1,40})[}\]]")


def tokenize(text: str) -> list[str]:
    r"""检索用 token 流：预切保护 span + jieba 细分 + 小写归一。

    公式文本（含 LaTeX）额外抽取 \command 与下标符号名 token。
    """
    tokens: list[str] = []
    if "\\" in text or "【公式】" in text or "【LaTeX】" in text:
        for m in _LATEX_CMD_RE.finditer(text):
            tokens.append(m.group(0).lower())
        for m in _LATEX_SYMNAME_RE.finditer(text):
            for w in jieba.lcut(m.group(1)):
                if w.strip():
                    tokens.append(w.strip().lower())
    for seg in precut(text):
        if _PRECUT_RE.fullmatch(seg):
            tokens.append(seg.replace(" ", "").lower())
            # 条款别名词元（双侧对称发射）：第十二条/第12条/12条 → §条12 跨形态互通
            try:
                from ..models.clause_norm import clause_alias_token
                alias = clause_alias_token(seg)
                if alias:
                    tokens.append(alias)
            except Exception:  # noqa: BLE001
                pass
            continue
        for w in jieba.lcut(seg):
            w = w.strip().lower()
            if w and not re.fullmatch(r"[\s\W]+", w, flags=re.UNICODE):
                tokens.append(w)
    return tokens
