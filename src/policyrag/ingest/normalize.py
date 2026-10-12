"""文本归一化：数字/日期/全角/中文数字双轨。

数字保真（计划 §五.3）的基础层：
- 归一化文本用于 ID 生成与校验比对；
- 同时维护 原文span ↔ 归一化span 的 offset 映射，保证引用可回溯原文。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "NormalizedText",
    "normalize",
    "extract_numeric_spans",
    "canonical_number",
    "canonical_date",
    "NUMERIC_SPAN_RE",
]

# ---------------------------------------------------------------------------
# 全角 → 半角（与 ids.normalize_text_for_id 同规则，但保留 offset 映射）
# ---------------------------------------------------------------------------

_FULLWIDTH_OFFSET = 0xFEE0


@dataclass
class NormalizedText:
    """原文 + 归一化文本 + offset 映射（orig[i] -> norm[j]）。"""

    orig: str
    norm: str
    # 对齐表：norm 的每个字符对应的 orig 起始下标
    norm_to_orig: list[int]

    def orig_span(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        """归一化区间 → 原文区间 [start, end)。"""
        if not self.norm_to_orig:
            return (0, 0)
        start = self.norm_to_orig[min(norm_start, len(self.norm_to_orig) - 1)]
        end = (
            self.norm_to_orig[norm_end - 1] + 1
            if 0 < norm_end <= len(self.norm_to_orig)
            else len(self.orig)
        )
        return (start, max(end, start))


def _to_halfwidth(ch: str) -> str:
    code = ord(ch)
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - _FULLWIDTH_OFFSET)
    if code == 0x3000:
        return " "
    return ch


def normalize(text: str) -> NormalizedText:
    """逐字符全角转半角 + NFC；空白折叠为单空格（保 offset）。"""
    norm_chars: list[str] = []
    mapping: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        h = _to_halfwidth(ch)
        h = unicodedata.normalize("NFC", h) if h != ch else h
        if h.isspace() or h == " ":
            if prev_space:
                continue
            norm_chars.append(" ")
            mapping.append(i)
            prev_space = True
        else:
            if len(h) == 1:
                norm_chars.append(h)
                mapping.append(i)
            else:  # NFC 偶发多字符展开，逐个映射到同一 orig 下标
                for sub in h:
                    norm_chars.append(sub)
                    mapping.append(i)
            prev_space = False
    return NormalizedText(orig=text, norm="".join(norm_chars), norm_to_orig=mapping)


# ---------------------------------------------------------------------------
# 数值 span 提取与规范化（数字保真比对用）
# ---------------------------------------------------------------------------

# 保护 span：日期、金额、百分比、中文百分数、文号、普通数字（含千分位、小数）
NUMERIC_SPAN_RE = re.compile(
    r"(?:"
    r"〔\d{2,4}〕\d+\s*号?"                              # 文号 〔2023〕1号
    r"|\d{4}\s*[-年/]\s*\d{1,2}\s*[-月/]\s*\d{1,2}\s*日?"  # 日期（容忍空格）
    r"|\d+\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"          # 2024 年 3 月 1 日
    r"|百分之[零〇一二两三四五六七八九十]+(?:点[零〇一二三四五六七八九十]+)?%?"  # 百分之八
    r"|\d[\d,]*\s*\.\s*\d+(?:\s+\d+)*\s*[%％]?"        # 断裂小数 1665 .13（表格抽取劈开，% 不丢）
    r"|\d+(?:\s*,\s*\d{3})+(?:\s*\.\s*\d+)?"           # 断裂千分位 1 ,400 / 19 ,455.5
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:万元|亿元|元|万|亿)?"  # 1,000.00 万元
    r"|\d+(?:\.\d+)?\s*(?:万元|亿元|万元?|%|％|个百分点|bp|基点|天|日|个月|年)"  # 带单位
    r"|\d+(?:\.\d+)?\s*%"                                # 百分比
    r"|\d+(?:\.\d+)?"                                    # 裸数字
    r")"
)

# 方向词规范化类（计划 §五.3：方向词与数字绑定校验）。
# 语义类 GE≥/LE≤/GT>/LT</UP↑/DOWN↓——"不得低于"与"不低于"同类（都 GE），
# "上调"与"下调"反向；答案的方向主张必须与 evidence 同类，防上调/下调混淆。
_DIR_PATTERNS: list[tuple[str, str]] = [
    (r"不(?:得|能|应)?(?:低于|少于|小于)", "GE"),
    (r"不(?:得|能|应)?(?:高于|多于|大于|超过)", "LE"),
    (r"不低于|不少于|至少|以上", "GE"),
    (r"不高于|不超过|不多于|至多|以下", "LE"),
    (r"上调|提高|增加|上涨|增长|上升", "UP"),
    (r"下调|降低|减少|下跌|下降|回落", "DOWN"),
    (r"低于|少于|小于", "LT"),
    (r"高于|多于|大于|超过", "GT"),
]
DIRECTION_RE = re.compile("|".join(p for p, _ in _DIR_PATTERNS))
_DIR_CLASS = [(re.compile(p), c) for p, c in _DIR_PATTERNS]


def canon_direction(window: str) -> str | None:
    """前文窗口 → 方向类（有序匹配：否定式优先于裸词）。"""
    for pat, cls in _DIR_CLASS:
        if pat.search(window):
            return cls
    return None

_CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    # 注意："十/百/千/万"是进位单位，必须走 _cn_int 的单位分支——
    # 曾误放数字表导致"十二"→2、"一百二十八"→108（中文数值全面失真）
}


def canonical_number(raw: str) -> str:
    """数值规范化：全角已半角；去千分位；中文数字→阿拉伯；统一单位形态。

    "百分之五十" -> "50%"；"1,000.00 万元" -> "1000.00万元"；"50％" -> "50%"
    """
    s = raw.strip().replace("％", "%").replace("　", " ")
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\s+", "", s)

    # 百分之X / 百分之X点Y
    m = re.fullmatch(r"百分之([零〇一二两三四五六七八九十]+)(?:点([零〇一二三四五六七八九十]+))?", s)
    if m:
        whole = _cn_int(m.group(1))
        if m.group(2):
            frac = "".join(str(_CN_NUM.get(c, c)) for c in m.group(2))
            return f"{whole}.{frac}%"
        return f"{whole}%"

    # 中文整数（简单区间，足够政策条款用）
    if re.fullmatch(r"[零〇一二两三四五六七八九十百千万]+", s):
        return str(_cn_int(s))

    # 数字与单位紧贴统一：去掉数字与单位间残留空格（normalize 已折叠）
    s = re.sub(r"(\d)(万元|亿元|元|万|亿|%|个百分点|天|日|个月|年)", r"\1\2", s)
    # 小数尾随零归一：2.40→2.4、40.0→40——同值不同形曾致保真闸误杀（表格 2.40 vs 答案 2.4）
    s = re.sub(r"\d+(?:\.\d+)?", lambda m: _strip_trail(m.group(0)), s)
    # 年份形态归一："2015年"/"2015E"（表格预测列头）→ "2015"——答案写年份、证据写 E 列
    if re.fullmatch(r"(?:19|20)\d{2}[年Ee]?", s):
        return s[:4]
    return s


def _strip_trail(num: str) -> str:
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def _cn_int(s: str) -> int:
    """简单中文数字 → int（一~万级，政策条款够用）。十/百/千/万按进位单位处理。"""
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in _CN_NUM:
            num = _CN_NUM[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
        elif ch == "万":
            total += (section + num) * 10000
            section, num = 0, 0
        else:
            return 0  # 无法解析，原样返回由调用方决定
    return total + section + num


_DATE_RE = re.compile(
    r"(\d{2,4})\s*[-年/]\s*(\d{1,2})\s*[-月/]\s*(\d{1,2})\s*日?"
)


def canonical_date(raw: str) -> str | None:
    """日期 → ISO (YYYY-MM-DD)。两位年视为 20xx。"""
    m = _DATE_RE.fullmatch(raw.strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


@dataclass
class NumericSpan:
    text: str          # 原文形态
    canonical: str     # 规范化形态（比对键）
    norm_start: int
    norm_end: int
    direction: str | None   # 方向类（GE/LE/GT/LT/UP/DOWN），来自左侧语境


# 引用/证据包装标记：数值提取一律屏蔽（防 UUID 碎片/[chunk 包装]/hex id 被当数值）
_CITE_MASK_RE = re.compile(
    r"\[[0-9a-fA-F]{6,}(?:-[0-9a-fA-F]{2,})+\]"   # UUID 形态 [10436a2e-7d3c-…]
    r"|\[[0-9a-fA-F]{16,64}\]"                     # 裸 hex id
    r"|\[chunk\s+[0-9a-fA-F]+[^\]]*\]"             # [chunk id …] 证据包装
)


def _mask_citation_marks(s: str) -> str:
    return _CITE_MASK_RE.sub(lambda m: " " * len(m.group(0)), s)


def extract_numeric_spans(nt: NormalizedText) -> list[NumericSpan]:
    """从归一化文本提取全部数值 span（含方向类绑定）。

    canonical 优先走 canonical_date（日期统一 ISO 双轨），
    再走 canonical_number；方向取左侧 6 字窗口的语义类。
    屏蔽：引用标记内部（UUID/hex/chunk 包装）；字母紧邻的数字（B2C→2、iPhone14 误提取）。
    """
    spans: list[NumericSpan] = []
    masked = _mask_citation_marks(nt.norm)
    for m in NUMERIC_SPAN_RE.finditer(masked):
        # 数字嵌在字母词中间（B2C/5G 型产品型号）不是数值
        if m.start() > 0:
            pre_ch = masked[m.start() - 1]
            if pre_ch.isascii() and pre_ch.isalpha():
                continue
        raw = m.group(0)
        # 日期双轨：与 evidence 侧统一为 ISO，防"2024-03-01" vs "2024 年 3 月 1 日"误判
        iso = canonical_date(raw)
        canonical = iso if iso else canonical_number(raw)
        pre = masked[max(0, m.start() - 6) : m.start()]
        spans.append(
            NumericSpan(
                text=raw,
                canonical=canonical,
                norm_start=m.start(),
                norm_end=m.end(),
                direction=canon_direction(pre),
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Symbol 字体私用区 → Unicode（PDF 数学符号常抽出为 U+F0xx；PoC 实测映射表）
# ---------------------------------------------------------------------------

SYMBOL_MAP = {
    0x61: "α", 0x62: "β", 0x64: "δ", 0x65: "ε", 0x66: "φ",
    0x67: "γ", 0x68: "η", 0x6C: "λ", 0x6D: "μ", 0x70: "π",
    0x71: "θ", 0x72: "ρ", 0x73: "σ", 0x74: "τ", 0x77: "ω",
    0x44: "Δ", 0x53: "Σ", 0x57: "Ω",
    0xE5: "∑", 0xD6: "∏", 0xE2: "∫", 0xB4: "×", 0xB8: "÷",
    0xB1: "±", 0x2D: "−", 0xA3: "≤", 0xB3: "≥", 0xB9: "≠",
    0xBB: "≈", 0xE4: "√", 0x22: "∞",
}


def map_symbol_pua(text: str) -> str:
    """U+F0xx 私用区字符映射回数学 Unicode；其余保留。"""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xF000 <= cp <= 0xF0FF:
            out.append(SYMBOL_MAP.get(cp - 0xF000, ch))
        else:
            out.append(ch)
    return "".join(out)
