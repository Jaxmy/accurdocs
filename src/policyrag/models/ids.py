"""稳定原子 ID 生成与解析。

ID 设计红线（见计划 §五.2）：
- 禁止把解析器版本/时间戳/顺序号编进 ID——否则重解析全库 ID 漂移，历史引用全部失效。
- 同一 (doc_id, version, clause_path, 规范化文本) 两次生成必须得到相同 ID（入库幂等）。
- text_norm = 全角转半角 + 空白折叠后的文本；原始文本另存。
- 解析质量升级导致 AST 变化时走显式 migration，不静默改 ID。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "ClausePath",
    "make_chunk_id",
    "make_doc_key",
    "normalize_text_for_id",
    "parse_chunk_id",
    "DOC_ID_RE",
]

# ---------------------------------------------------------------------------
# 孤立代理字符清理（Windows 常见：浏览器 multipart 文件名 / PDF 抽取文本）
# Python str 里的 U+D800-DFFF 无法 encode('utf-8')，必须在边界清理。
# ---------------------------------------------------------------------------

_SURRO_RE = re.compile("[\ud800-\udfff]")


def desurrogate(s: str) -> str:
    """清除/还原孤立代理字符。

    DC80-DCFF 是 surrogateescape 的字节残留，先尝试还原原始字节
    （UTF-8 → GBK 两级回退）；其余高位代理直接剔除。无代理时零开销原样返回。
    """
    if not _SURRO_RE.search(s):
        return s
    try:
        raw = s.encode("utf-8", errors="surrogateescape")
        for enc in ("utf-8", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    except UnicodeEncodeError:
        # 高位代理（U+D800 等）无法经 surrogateescape 还原 → 剔除
        return "".join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))


# 文号/稳定短名：允许中文、〔〕、数字、字母、点号、连字符、下划线
DOC_ID_RE = re.compile(r"^[一-鿿A-Za-z0-9_〔〕()\-\[\].]{1,120}$")


def sanitize_doc_id(raw: str) -> str:
    """文件名/用户输入 → 合法 doc_id（确定性、幂等）。

    真实文件名含全角括号（）、《》、空格、点号等——不能拒收，要清洗：
    全角转半角 + 去空白 + 剔除不允许字符。空结果时用原名哈希保证唯一。
    """
    s = normalize_text_for_id(raw)  # 去代理 + 全角转半角 + 去所有空白
    s = re.sub(r"[^一-鿿A-Za-z0-9_〔〕()\-\[\].]", "", s)
    if not s:
        import hashlib as _h
        s = "doc_" + _h.sha1(desurrogate(raw).encode("utf-8", "replace")).hexdigest()[:8]
    return s[:120]

_FULLWIDTH_OFFSET = 0xFEE0  # ！(FF01) -> !(21)；数字/字母同理


def normalize_text_for_id(text: str) -> str:
    """ID 用规范化文本：去代理字符 + 全角转半角 + 去除全部空白。确定性纯函数。

    空白在中文法规文本中无语义，彻底去除可保证跨渲染（"1,000 万元"/"1,000万元"）
    ID 稳定，并让版本 diff 的规范化文本 hash 全等对齐不被空白差异干扰。
    """
    out: list[str] = []
    for ch in desurrogate(text):
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII 区
            out.append(chr(code - _FULLWIDTH_OFFSET))
        elif code == 0x3000:  # 全角空格
            continue
        elif ch.isspace():
            continue
        else:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class ClausePath:
    """章/节/条/款/项/表 的层级路径。

    key 是确定性字符串，直接进 ID 哈希。
    """

    parts: tuple[str, ...]

    @staticmethod
    def of(*parts: str) -> "ClausePath":
        cleaned = tuple(p.strip() for p in parts if p and p.strip())
        return ClausePath(cleaned)

    @property
    def key(self) -> str:
        return "/".join(self.parts)

    @property
    def display(self) -> str:
        return " ".join(self.parts)

    def __str__(self) -> str:  # pragma: no cover
        return self.key


def make_doc_key(doc_id: str, version: str) -> str:
    """文档-版本键（registry 主键的一部分）。"""
    return f"{doc_id}@{version}"


def make_chunk_id(doc_id: str, version: str, path: ClausePath, text_norm: str) -> str:
    """稳定原子 chunk ID：blake2s(doc_id|version|path.key|sha1(text_norm))[:16]。

    - text_norm 必须先经 normalize_text_for_id。
    - 输出 16 个 hex 字符，引用短、可复制、可校验。
    """
    if not DOC_ID_RE.match(doc_id):
        # 防御：入口已 sanitize_doc_id；此处仍非法说明调用方绕过了入口
        raise ValueError(f"doc_id 含非法字符（应先经 sanitize_doc_id）: {doc_id!r}")
    inner = hashlib.sha1(text_norm.encode("utf-8")).hexdigest()
    material = f"{doc_id}|{version}|{path.key}|{inner}"
    return hashlib.blake2s(material.encode("utf-8"), digest_size=8).hexdigest()


def parse_chunk_id(chunk_id: str) -> None:
    """校验 chunk_id 格式（16 hex）。ID 本身不含结构信息——结构查 registry。"""
    if not re.fullmatch(r"[0-9a-f]{16}", chunk_id):
        raise ValueError(f"非法 chunk_id: {chunk_id!r}")


def make_rich_chunk_id(doc_id: str, version: str, kind: str, local_id: str) -> str:
    """公式/图表块稳定 ID：只依赖 (doc, version, kind, local_id)。

    与 make_chunk_id 不同——富化块的身份是版面位置（页+序号），
    转写内容从占位升级为 LaTeX 时 ID 不变，引用与缓存不失效。
    """
    if not DOC_ID_RE.match(doc_id):
        raise ValueError(f"doc_id 含非法字符（应先经 sanitize_doc_id）: {doc_id!r}")
    material = f"{doc_id}|{version}|{kind}|{local_id}"
    return hashlib.blake2s(material.encode("utf-8"), digest_size=8).hexdigest()
