"""文档身份治理（WS-E）：标题提取链 + 内容指纹 + 近似判定 + 版本组。

设计约束：
- 存量 doc_id/doc_key 冻结（make_chunk_id 含 path.key 哈希——ID 稳定红线）；
  本模块只服务于**新入库**的身份判定与显示字段。
- 近似判定确定性零依赖：content_hash 全等 / 64 位 simhash / SequenceMatcher。
- 标题提取链内容优先（上传路径曾用文件名 stem 短路提取——G8 已修）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = [
    "extract_title", "content_fingerprint", "simhash64", "hamming",
    "title_key", "title_similar", "clean_filename_stem", "DocIdentity",
    "extract_version_markers", "base_title",
]

# ---------------------------------------------------------------- 标题提取

_BOOK_TITLE_RE = re.compile(r"《([^》\n]{2,80})》")
_TITLE_TYPE_RE = re.compile(
    r".{2,40}(办法|规定|通知|细则|条例|意见|公告|方案|准则|指引|规则)"
    r"\s*(?:（[^）]{0,24}）|\([^)]{0,24}\))?\s*$"
)
_DOC_ENTER_RE = re.compile(r"^第[一二三四五六七八九十百千零〇\d０-９]+[章节条]")

# 垃圾行过滤（页码装饰/水印/纯符号/表格残迹/重复字符）
_GARBAGE_RES = [
    re.compile(r"^[—\-–—\s]*\d{1,4}[—\-–—\s]*$"),          # — 93 —
    re.compile(r"^第\s*\d{1,3}\s*页"),
    re.compile(r"^\d{4}\s*年\s*\d{0,2}\s*月?$"),           # 2025 年月
    re.compile(r"^[\d\s\W_]+$"),                            # 纯数字/符号
    re.compile(r"(下载|扫描件|仅供|内部学习|请勿外传)"),       # 水印词
    re.compile(r"[\t]|\s{3,}"),                             # 表格残迹
    re.compile(r"[\ud800-\udfff-]"),           # 代理/PUA 残留
    re.compile(r"(.)\1{3,}"),                               # 重复字符 ≥4
]


def _ok_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    for rx in _GARBAGE_RES:
        if rx.search(s):
            return False
    if len(s) < 6 and "《" not in s and not _TITLE_TYPE_RE.search(s):
        return False
    return True


def clean_filename_stem(stem: str) -> str:
    """文件名兜底清洗：去 sha1 前缀 / 副本后缀 / 全角空格折叠。"""
    s = re.sub(r"^[0-9a-f]{8,}", "", stem)                  # sha1 前缀
    s = re.sub(r"[-_ ]*副本\s*\d*$", "", s)
    s = re.sub(r"\(\d+\)$", "", s)
    s = re.sub(r"（\d+）$", "", s)
    s = s.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:120] or stem[:120]


def extract_title(raw_text: str, filename_stem: str) -> tuple[str, str]:
    """内容优先标题链。返回 (title, source)。

    ①《...》书名号 ②文号/类型后缀题名行（第一章前）③首个过垃圾过滤行（≤5 行）
    ④文件名清洗兜底。
    """
    lines = [re.sub(r"^#+\s*", "", ln.strip()) for ln in raw_text.splitlines()
             if ln.strip()][:30]

    m = _BOOK_TITLE_RE.search(raw_text[:3000])
    if m and _ok_line(m.group(1)):
        return m.group(1).strip()[:120], "book_title"

    for ln in lines:
        if _DOC_ENTER_RE.match(ln):
            break                                    # 已进入正文，题名区结束
        if _TITLE_TYPE_RE.search(ln) and _ok_line(ln):
            return ln[:120], "type_suffix"

    tried = 0
    for ln in lines:
        tried += 1
        if tried > 5:
            break
        if _ok_line(ln):
            return ln[:120], "first_line"

    return clean_filename_stem(filename_stem), "filename"


# ---------------------------------------------------------------- 内容指纹

def content_fingerprint(text: str) -> str:
    """全文归一化指纹（同文重传短路判定）。"""
    from ..models.ids import normalize_text_for_id
    return hashlib.sha1(normalize_text_for_id(text).encode("utf-8")).hexdigest()[:16]


def simhash64(text: str) -> int:
    """64 位 simhash（字符 2-gram 异或折叠）——近似候选预筛，纯 Python。"""
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 2:
        return 0
    vec = [0] * 64
    for i in range(len(t) - 1):
        g = t[i:i + 2]
        h = int.from_bytes(hashlib.sha1(g.encode("utf-8")).digest()[:8], "big")
        for b in range(64):
            vec[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if vec[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------- 标题相似

_PUNCT_RE = re.compile(r"[\s《》〈〉「」“”\"'（）()\[\]【】，,。.、：:；;—\-–·\d]")


def title_key(title: str) -> str:
    """标题归一键（去书名号/标点/数字/空白）。"""
    return _PUNCT_RE.sub("", title or "")


def title_similar(a: str, b: str) -> float:
    ka, kb = title_key(a), title_key(b)
    if not ka or not kb:
        return 0.0
    return SequenceMatcher(None, ka, kb).ratio()


# ---------------------------------------------------------------- 版本标记（T0 快判）

# 政策：版本号/稿态/修订；研报财报：财年+季度/报告类型（用户决策：按标题索引先判版本）
_DOC_NO_MARK_RE = re.compile(r"([一-鿿]{2,8}〔\d{2,4}〕\d+\s*号)")
_VERSION_MARK_RE = re.compile(
    r"第[一二三四五六七八九十百零〇\d]+版"           # 第二版
    r"|[Vv]\d+(?:\.\d+){1,3}"                        # v2.1
    r"|(?:版本|版次)[号：:\s]*[\w.\-]{1,12}"          # 版本号：2023
    r"|\d{4}\s*年\s*修订"                            # 2023年修订
)
_DRAFT_STATE_RE = re.compile(
    r"(征求意见稿|送审稿|报批稿|草案|试行|暂行|正式稿|发布稿|定稿)")
_PERIOD_RE = re.compile(
    r"(\d{4})\s*年?\s*([一二三四1-4])\s*季度(?:报告|报)?"      # 2024年三季度
    r"|(\d{4})\s*年?\s*[Qq]([1-4])"                            # 2024Q3
    r"|(\d{4})\s*年?\s*(三季|半年度?|中期|年度)(?:报告|报)?"    # 三季报/半年度/年度
)
_Q_CN = {"一": "1", "二": "2", "三": "3", "四": "4"}


def extract_version_markers(text: str, title: str = "") -> dict:
    """T0 标题/首页快判：版本标记提取。政策→文号/版本号/稿态；研报财报→财年季度。

    只看题名区（标题 + 文本前 4000 字）——先于全文确判（T1）给出版本信号。
    """
    head = f"{title or ''}\n{(text or '')[:4000]}"
    out: dict = {"doc_no": None, "version_marker": None, "draft_state": None,
                 "period": None}
    m = _DOC_NO_MARK_RE.search(head)
    if m:
        out["doc_no"] = m.group(1).replace(" ", "")
    m = _VERSION_MARK_RE.search(head)
    if m:
        out["version_marker"] = re.sub(r"\s+", "", m.group(0))[:24]
    m = _DRAFT_STATE_RE.search(head)
    if m:
        out["draft_state"] = m.group(1)
    m = _PERIOD_RE.search(head)
    if m:
        g = m.groups()
        if g[0] and g[1]:
            out["period"] = f"{g[0]}Q{_Q_CN.get(g[1], g[1])}"
        elif g[2] and g[3]:
            out["period"] = f"{g[2]}Q{g[3]}"
        elif g[4] and g[5]:
            kind = g[5]
            out["period"] = (f"{g[4]}Q3" if kind == "三季"
                             else f"{g[4]}H1" if kind in ("半年", "半年度", "中期")
                             else f"{g[4]}年度")
    out["has_marker"] = any(out[k] for k in
                            ("doc_no", "version_marker", "draft_state", "period"))
    return out


# 基名剥离：稿态/版本/期间标记（title_key 再去标点数字）
_MARKER_STRIP_RE = re.compile(
    r"征求意见稿|送审稿|报批稿|草案|试行|暂行|正式稿|发布稿|定稿"
    r"|第[一二三四五六七八九十百零〇\d]+版|[Vv]\d+(?:\.\d+){1,3}"
    r"|\d{4}\s*年?\s*(?:[一二三四1-4]\s*季度|[Qq][1-4]|三季|半年度?|中期|年度)"
    r"(?:报告|报|修订)?"
    r"|\d{4}\s*年?\s*修订"
)


def base_title(title: str) -> str:
    """标题基名（剥版本/稿态/期间标记）——同基名=同一文档产品（版本演进不拆族）。"""
    return title_key(_MARKER_STRIP_RE.sub("", title or ""))


def _markers_differ(a: dict, b: dict) -> bool:
    """版本演进信号：任一标记字段不同（版本号/稿态/期间/文号）。"""
    return any(a.get(k) and b.get(k) and a.get(k) != b.get(k)
               for k in ("doc_no", "version_marker", "draft_state", "period"))


def _row_markers(row: dict) -> dict:
    """存量行版本标记：meta_json.version_markers（新写入）> 标题提取。"""
    import json as _j
    try:
        meta = _j.loads(row.get("meta_json") or "{}")
    except Exception:  # noqa: BLE001
        meta = {}
    m = meta.get("version_markers")
    if isinstance(m, dict) and m:
        m.setdefault("has_marker", any(m.get(k) for k in
                    ("doc_no", "version_marker", "draft_state", "period")))
        return m
    return extract_version_markers("", row.get("title") or "")


# ---------------------------------------------------------------- 身份判定

@dataclass
class DocIdentity:
    doc_group: str          # 身份组（组内最早 doc_id，稳定）
    version: str            # 最终版本标签（含 -r{n} 防覆盖后缀）
    version_seq: int
    is_latest: bool
    content_hash: str
    near_dup_of: str | None = None   # 与哪份已有文档近似（供 meta 记录）
    markers: dict | None = None      # T0 版本标记（标题/首页快判）
    relation: str = "new"            # new / same_doc_version / near_dup


def decide_identity(
    *,
    doc_id: str,
    version: str,
    title: str,
    doc_no: str | None,
    raw_text: str,
    existing: list[dict],
) -> DocIdentity:
    """新文档 vs 存量文档 → 身份组 / 版本号 / 是否最新。

    existing: documents 行（dict，含 doc_key/doc_id/version/title/doc_group/
    version_seq/content_hash/created_at）。
    近似终判：标题相似 ≥0.9 **且**（同文号 | simhash≤6 | 前 20k 字相似 ≥0.9）；
    T0 版本标记强信号：同基名 + 双侧有版本标记 → 同族（大改版也归组，不拆两文档）。
    """
    chash = content_fingerprint(raw_text)
    sim = simhash64(raw_text[:20000])
    markers = extract_version_markers(raw_text[:4000], title)
    base_new = base_title(title)

    # 近似终判：标题相似 ≥0.9 且（同文号 | simhash≤6 | 指纹全等）——单靠标题不成立；
    # 但同基名 + 双侧版本标记（版本号/文号/稿态/财年季度）= 同一文档产品的版本演进
    best: dict | None = None
    best_score = 0.0
    best_marker_family = False
    for row in existing:
        same_no = bool(doc_no) and _row_doc_no(row) == doc_no
        t_sim = title_similar(title, row.get("title") or "")
        row_markers = _row_markers(row)
        base_sim = (SequenceMatcher(None, base_new,
                                    base_title(row.get("title") or "")).ratio()
                    if base_new else 0.0)
        marker_family = (base_sim >= 0.9 and len(base_new) >= 4
                         and markers.get("has_marker")
                         and row_markers.get("has_marker"))
        if t_sim < 0.9 and not same_no and not marker_family:
            continue
        row_hash = row.get("content_hash") or ""
        row_simhash = row.get("simhash") or 0
        same_family = (same_no or marker_family or row_hash == chash
                       or (row_simhash and hamming(sim, int(row_simhash)) <= 6))
        if same_family:
            score = max(t_sim, base_sim if marker_family else 0.0) \
                + (0.5 if same_no else 0.0) + (1.0 if row_hash == chash else 0.0) \
                + (0.5 if marker_family and _markers_differ(markers, row_markers) else 0.0)
            if score > best_score:
                best, best_score = row, score
                best_marker_family = marker_family

    if best is None:
        return DocIdentity(doc_group=doc_id, version=version, version_seq=1,
                           is_latest=True, content_hash=chash,
                           markers=markers, relation="new")

    group = best.get("doc_group") or best.get("doc_id") or doc_id
    seqs = [r.get("version_seq") or 1 for r in existing
            if (r.get("doc_group") or r.get("doc_id")) == group]
    seq = (max(seqs) if seqs else 1) + 1
    relation = ("same_doc_version" if best_marker_family
                else "near_dup")
    return DocIdentity(doc_group=group, version=version, version_seq=seq,
                       is_latest=True, content_hash=chash,
                       near_dup_of=best.get("doc_key"),
                       markers=markers, relation=relation)


def resolve_doc_key(meta, doc_id: str, version: str, chash: str) -> tuple[str, bool]:
    """doc_key 冲突防覆盖：同 key 异指纹 → 版本后缀 -r{n}；同指纹 → noop。

    返回 (final_version, is_noop)。
    """
    base = version
    n = 1
    while True:
        v = base if n == 1 else f"{base}-r{n}"
        row = meta.get_document(f"{doc_id}@{v}")
        if row is None:
            return v, False
        if (row.get("content_hash") or "") == chash:
            return v, True
        n += 1


def _row_doc_no(row: dict) -> str | None:
    import json as _j
    try:
        meta = _j.loads(row.get("meta_json") or "{}")
    except Exception:  # noqa: BLE001
        return None
    return meta.get("doc_no")
