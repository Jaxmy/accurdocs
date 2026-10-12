"""条款树 AST 构建：中文政策/法规结构识别。

识别层级：章 → 节 → 条 → 款 → 项 → 表。
红线（计划 §五.1）：
- 切分前数字/日期/金额/文号先打保护 span，禁 span 内断开；
- "但书/除外"句与本款同块；
- 条款号识别失败 → 段落级 path 并告警（不硬猜）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.document import BlockKind, DocNode, TableBlock
from ..models.ids import ClausePath

__all__ = ["build_ast", "StructureWarning"]

# --- 结构行正则（先保护 span，再按行首标记识别） ---------------------------

RE_CHAPTER = re.compile(r"^第([一二三四五六七八九十百千零〇\d]+)章\s*(.*)$")
RE_SECTION = re.compile(r"^第([一二三四五六七八九十百千零〇\d]+)节\s*(.*)$")
RE_ARTICLE = re.compile(r"^第([一二三四五六七八九十百千零〇\d]+)条(之[一二三四五六七八九十\d]+)?\s*(.*)$")
RE_PARAGRAPH = re.compile(r"^第([一二三四五六七八九十百千零〇\d]+)款\s*(.*)$")
RE_ITEM = re.compile(r"^[（(]([一二三四五六七八九十\d]+)[）)]\s*(.*)$")
RE_CN_NUM = re.compile(r"^[一二三四五六七八九十百千零〇\d]+[、.．]\s*(.*)$")

# 但书/除外续行标记：与上句同块，不新起款
_BUT_CONT_RE = re.compile(r"^(但\s*书|但是|但，|但:|除[此该本前述][^。]{0,40}外)")

# 保护 span：文号/日期/金额/百分比——结构识别时不允许这些内部被当成换行切点
PROTECT_RE = re.compile(
    r"〔\d{2,4}〕\d+\s*号"
    r"|\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*(?:万元|亿元|元|%|％|个百分点)"
)

# --- 标题/噪声行识别（索引破碎根因治理） -----------------------------------
# 标题行：短行 + 无句读 + 标题型后缀（政策分则名、研报栏目名通用词面）。
# 曾无此识别：总标题碎成多行各自成节点、分则/栏目标题不成节点、正文行头
# 以 text[:12] 混进索引。
_TITLE_SUFFIX_RE = re.compile(
    r"(实施细则|实施方案|管理办法|暂行规定|基本规则|运营规则|交易规则"
    r"|规则|细则|办法|规定|通知|意见|方案|指引|准则|条例|公告|要点|大纲"
    r"|观点|主题|引言|前言|摘要|概述|综述|结论|总结|展望|风险提示|投资建议"
    r"|附录|附件|总则|附则|名词解释|行业|报告)$"
)
# 标题续行前缀：（试行…稿）/ 及配套实施细则 等封面碎行并回标题
_CONT_PREFIX = "（(至及与和或"

# 目录行：点线引导（曾被 RE_CHAPTER 认成真章条 → 幽灵结构、序错乱）
_TOC_LINE_RE = re.compile(r"\.{3,}|…{2,}")

# 噪声行：页脚页码/裸年月/纯数据行/纯符号/单字残片（不进结构流）
_GARBAGE_LINE_RE = re.compile(
    r"^[—\-–—\s]*\d{1,4}[—\-–—\s]*$"          # — 3 —
    r"|^第\s*\d{1,3}\s*页"                    # 第2页
    r"|^目\s*录$"                             # 目录页标（非内容标题）
    r"|^\d{4}\s*年\s*\d{0,2}\s*月?\s*$"       # 2025 年月
    r"|^[\d\s.%％,，Xx万亿]+$"                 # 纯数据行 1% / 1,992 / 108X
    r"|^[·•|｜/\\—\-–—\s]+$"                  # 纯符号
)
_SENT_END_RE = re.compile(r"[。！？；;]$")


def _is_noise_line(t: str) -> bool:
    """目录点线行/页脚页码/纯数据/单字残片——不进结构流。"""
    return (len(t) <= 1 or bool(_GARBAGE_LINE_RE.match(t))
            or bool(_TOC_LINE_RE.search(t)))


def _is_heading_line(t: str) -> bool:
    """标题行：短、无句读、非编号开头、命中标题型后缀。"""
    return (2 <= len(t) <= 40
            and not re.search(r"[。；;！!？?，,：:]", t)
            and not re.match(r"^[0-9（(【\[]", t)
            and bool(_TITLE_SUFFIX_RE.search(t)))


# 编号块起始行（名词解释"4.xxx："、条文列举"（三）"、中文序号"一、"）——
# 命中即另起新块，不并入上一块
_NUM_START_RE = re.compile(
    r"^\d{1,3}\s*[.、．]"
    r"|^[（(][一二三四五六七八九十\d]{1,3}[）)]"
    r"|^[一二三四五六七八九十]{1,3}[、.．]"
)


def _append_body_line(siblings: list, t: str, page: int | None) -> None:
    """标题下/根层正文行落块：排版折行并回上块（"条目即原子块"）。

    曾每行各自成块——名词解释条目被切成 25 字碎块（"4.电力现货市场…：通过交易平台在"
    截断，两子市场名落入邻块），检索拿到碎块必拒答。合并规则与条款侧一致：
    上一行未句末收尾 且 本行不是新编号块 → 并回；否则另起 TEXT 块。
    """
    if (siblings and siblings[-1].kind == BlockKind.TEXT
            and not _NUM_START_RE.match(t)
            and not _SENT_END_RE.search(siblings[-1].text)):
        siblings[-1].text = (siblings[-1].text + " " + t).strip()
    else:
        siblings.append(
            DocNode(kind=BlockKind.TEXT, number="", text=t, page=page))


@dataclass
class StructureWarning:
    kind: str
    detail: str
    line_no: int


@dataclass
class _RawLine:
    text: str
    page: int | None = None
    is_table_row: bool = False
    table_rows: list[list[str]] = field(default_factory=list)
    table_caption: str = ""
    table_headers: list[str] = field(default_factory=list)


def _split_lines(pages: list[tuple[str, int | None]]) -> list[_RawLine]:
    """把 (page_text, page_no) 列表摊平为行；表格块不拆行。"""
    out: list[_RawLine] = []
    for text, page in pages:
        for ln in text.splitlines():
            ln = ln.rstrip()
            if ln.strip():
                out.append(_RawLine(text=ln, page=page))
    return out


def build_ast(
    pages: list[tuple[str, int | None]],
    tables: list[TableBlock] | None = None,
) -> tuple[list[DocNode], list[StructureWarning]]:
    """行流 → 条款树。返回 (nodes, warnings)。

    pages: [(page_text, page_no), ...]；tables: 已结构化的表格（附在最近节点）。
    """
    warnings: list[StructureWarning] = []
    root_nodes: list[DocNode] = []
    cur_chapter: DocNode | None = None
    cur_section: DocNode | None = None
    cur_article: DocNode | None = None
    cur_para: DocNode | None = None
    cur_heading: DocNode | None = None   # 总标题/分则标题/栏目标题
    chapter_no = section_no = article_no = para_no = item_no = 0
    fallback_mode = False
    body_started = False   # 当前标题下是否已有正文（区分标题续行）
    seen_content = False   # 题名区（文档首个内容行）判定

    lines = _split_lines(pages)

    prev_open = False       # 上一内容行未以句末收尾（当前行是排版折行续行）
    prev_page: int | None = None
    line_in_page = 0

    for i, line in enumerate(lines):
        t = re.sub(r"^#+\s*", "", line.text.strip())
        if not t or _is_noise_line(t):
            continue
        if line.page != prev_page:
            prev_page = line.page
            line_in_page = 0
            prev_open = False      # 翻页即新语境（跨页折行误判为标题的代价更低）
        line_in_page += 1
        page_top = line_in_page <= 3
        cont_wrap = prev_open      # 本行是否上一行的排版折行续行
        # 折行语境：句末自然闭合；逗号类收尾或长行才算"未收尾"——
        # 短而无句读的行（标题/标签）是闭合语境，后继行可起新标题
        prev_open = (not _SENT_END_RE.search(t)
                     and (len(t) > 20 or bool(re.search(r"[，,、；]$", t))))

        # 章（挂当前标题下：分则标题 → 章节条 形成完整层级）
        m = RE_CHAPTER.match(t)
        if m and len(t) < 60:
            chapter_no += 1
            cur_chapter = DocNode(
                kind=BlockKind.CHAPTER, number=f"第{m.group(1)}章",
                text=m.group(2).strip(), page=line.page,
            )
            (cur_heading.children if cur_heading else root_nodes).append(cur_chapter)
            cur_section = cur_article = cur_para = None
            body_started = seen_content = True
            continue

        # 节
        m = RE_SECTION.match(t)
        if m and len(t) < 60 and cur_chapter is not None:
            section_no += 1
            cur_section = DocNode(
                kind=BlockKind.SECTION, number=f"第{m.group(1)}节",
                text=m.group(2).strip(), page=line.page,
            )
            (cur_chapter.children if cur_chapter else root_nodes).append(cur_section)
            cur_article = cur_para = None
            body_started = seen_content = True
            continue

        # 条
        m = RE_ARTICLE.match(t)
        if m:
            suffix = m.group(2) or ""
            article_no += 1
            cur_article = DocNode(
                kind=BlockKind.ARTICLE,
                number=f"第{m.group(1)}条{suffix}",
                text=m.group(3).strip(),
                page=line.page,
            )
            parent = cur_section or cur_chapter or cur_heading
            if parent is not None:
                parent.children.append(cur_article)
            else:
                root_nodes.append(cur_article)
            cur_para = None
            item_no = 0
            body_started = seen_content = True
            continue

        # 款（独立成行的"第X款"或无编号换行段落归入当前条）
        m = RE_PARAGRAPH.match(t)
        if m and cur_article is not None:
            para_no += 1
            cur_para = DocNode(
                kind=BlockKind.PARAGRAPH, number=f"第{m.group(1)}款",
                text=m.group(2).strip(), page=line.page,
            )
            cur_article.children.append(cur_para)
            item_no = 0
            body_started = seen_content = True
            continue

        # 项 （一）（二）
        m = RE_ITEM.match(t)
        if m and (cur_para is not None or cur_article is not None):
            item_no += 1
            item = DocNode(
                kind=BlockKind.ITEM, number=f"（{m.group(1)}）",
                text=m.group(2).strip(), page=line.page,
            )
            parent = cur_para or cur_article
            assert parent is not None
            parent.children.append(item)
            body_started = seen_content = True
            continue

        # 但书/除外续行：与本款（或本条）同块，不新起款（计划 §五.1 红线）
        if _BUT_CONT_RE.match(t):
            if cur_para is not None:
                cur_para.text = (cur_para.text + " " + t).strip()
            elif cur_article is not None:
                cur_article.text = (cur_article.text + " " + t).strip()
            elif root_nodes:
                last = root_nodes[-1]
                last.text = (last.text + " " + t).strip()
            continue

        # 标题续行：当前标题尚无正文 +（试行…稿）/及配套… 等封面碎行并回标题
        if (cur_heading is not None and not body_started
                and t[0] in _CONT_PREFIX and len(t) <= 30):
            cur_heading.text = (cur_heading.text + t).strip()
            continue

        # 标题行：短行+无句读+标题后缀（政策分则名/研报栏目名）。
        # 准入：非折行续行 且（页首 3 行 或 条/款正文流之外）——
        # 折行续行"按照本规则"不误判；多分则标题（页首、上一分则条流之后）必识别
        if (_is_heading_line(t) and not cont_wrap
                and (page_top or (cur_article is None and cur_para is None))):
            cur_heading = DocNode(
                kind=BlockKind.HEADING, number="", text=t, page=line.page,
            )
            root_nodes.append(cur_heading)
            cur_section = cur_article = cur_para = None
            body_started = False
            seen_content = True
            continue

        # 题名区首行：文档首个内容行且短而无句读 → 总标题（无后缀也认）
        if not seen_content and len(t) <= 50 and not re.search(r"[。！？；;]", t):
            seen_content = True
            cur_heading = DocNode(
                kind=BlockKind.HEADING, number="", text=t, page=line.page,
            )
            root_nodes.append(cur_heading)
            continue
        seen_content = True

        # 兜底：续行并入最近节点；无任何结构锚点 → 段落级 fallback
        body_started = True
        # 折行续行优先并入最近的开放子块（项/款末尾折行"…应提前45个"+"工作日…"
        # 曾各自成块——条款/项即原子块，排版折行不是新块）
        merged = False
        if not _NUM_START_RE.match(t):
            for parent in (cur_para, cur_article):
                if parent is not None and parent.children:
                    last = parent.children[-1]
                    if (last.kind in (BlockKind.ITEM, BlockKind.PARAGRAPH)
                            and last.text and not _SENT_END_RE.search(last.text)):
                        last.text = (last.text + " " + t).strip()
                        merged = True
                        break
        if merged:
            pass
        elif cur_para is not None:
            cur_para.text = (cur_para.text + " " + t).strip()
        elif cur_article is not None:
            if cur_article.text and not _SENT_END_RE.search(cur_article.text):
                # 换行续行并回本条（"条款即原子块"——排版折行不是新款）
                cur_article.text = (cur_article.text + " " + t).strip()
            else:
                para_no += 1
                cur_para = DocNode(
                    kind=BlockKind.PARAGRAPH, number="", text=t, page=line.page,
                )
                cur_article.children.append(cur_para)
        elif cur_heading is not None:
            _append_body_line(cur_heading.children, t, line.page)
        else:
            if not fallback_mode:
                fallback_mode = True
                warnings.append(StructureWarning(
                    kind="no_structure_anchor",
                    detail="文档缺少章/条结构，降级为段落级 path",
                    line_no=i,
                ))
            _append_body_line(root_nodes, t, line.page)

    # 未闭合检查
    if root_nodes and all(n.kind == BlockKind.TEXT for n in root_nodes):
        warnings.append(StructureWarning(
            kind="degraded_layout", detail="全部节点为 TEXT，条款识别失败", line_no=0,
        ))

    # 表格按页码就近挂载（曾全部挂到根尾节点 → 表块归属错位）
    if tables:
        all_nodes = [n for r in root_nodes for n in r.walk()]
        for tb in tables:
            tnode = DocNode(kind=BlockKind.TABLE, number="", text=tb.caption, page=tb.page)
            tnode.meta["table"] = tb
            anchor = min(
                (n for n in all_nodes if n.kind != BlockKind.TABLE),
                key=lambda n: abs((n.page or 0) - (tb.page or 0)), default=None)
            if anchor is not None:
                anchor.children.append(tnode)
            else:
                root_nodes.append(tnode)

    _assign_paths(root_nodes)
    return root_nodes, warnings


def _assign_paths(nodes: list[DocNode], prefix: tuple[str, ...] = ()) -> None:
    """路径段纪律（索引破碎根因）：只有结构编号与真标题占段。

    正文行/无编号款/表块的 text[:12] 曾直接当路径段——正文行头、人名、
    表格数字全成了索引项。现在正文不占段（path=父链），标题用完整短标题。
    """
    for node in nodes:
        if node.kind == BlockKind.HEADING:
            # 路径段去内部空白——display 以空格连接段，段内空格会把标题劈碎
            seg = re.sub(r"\s+", "", node.text)[:40]
        elif node.kind in (BlockKind.TEXT, BlockKind.TABLE):
            seg = ""                       # 正文/表块不占路径段
        elif node.kind == BlockKind.PARAGRAPH and not node.number:
            seg = ""                       # 无编号款同样不占段
        else:
            seg = node.number or node.text[:12]
        path = ClausePath.of(*prefix, seg)
        node.path = path
        _assign_paths(node.children, prefix=path.parts)
