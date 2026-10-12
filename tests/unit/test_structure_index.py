"""结构识别/索引树回归：索引破碎与意义不明的机制修复。

病灶（现场实测）：目录点线行被认成真章条（幽灵结构）；总标题碎三行各自成
节点；分则/栏目标题不成节点；正文行头以 text[:12] 当路径段（人名/表格数
字/页脚全进索引）；条款排版折行被拆成新款。
"""

from policyrag.ingest.structure import build_ast
from policyrag.models.document import BlockKind


def _paths(nodes):
    out = []

    def walk(ns):
        for n in ns:
            out.append((n.kind, n.path.display))
            walk(n.children)
    walk(nodes)
    return out


def test_cover_title_fragments_merge():
    """封面三碎行 → 一个标题节点；年月噪声行不进。"""
    pages = [(
        "辽宁省电力市场运营规则\n及配套实施细则\n（试行4.0 版征求意见稿）\n2025 年月\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    assert len(nodes) == 1, _paths(nodes)
    h = nodes[0]
    assert h.kind == BlockKind.HEADING
    assert h.text == "辽宁省电力市场运营规则及配套实施细则（试行4.0 版征求意见稿）"


def test_toc_lines_not_phantom_structure():
    """目录点线行（第一章总则....）不产章条——曾制造幽灵结构、序错乱。"""
    pages = [(
        "辽宁省电力市场运营规则\n"
        "目录\n"
        "第一章总则.......................................................\n"
        "第一条为规范市场运营，制订本规则。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    kinds = [n.kind for n in nodes]
    assert BlockKind.CHAPTER not in kinds, _paths(nodes)   # 目录点线章不产章节点
    arts = [n for n in nodes[0].walk() if n.kind == BlockKind.ARTICLE]
    assert len(arts) == 1 and arts[0].number == "第一条"   # 真条仍识别


def test_subdoc_title_headings_nest_chapters():
    """分则标题成节点且章挂其下（总标题/分则标题即索引层级）。"""
    pages = [(
        "辽宁省电力市场运营规则\n"
        "辽宁省现货电能量市场交易实施细则\n"
        "第一章总则\n"
        "第一条为规范现货市场，制订本细则。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    assert nodes[0].kind == BlockKind.HEADING
    sub = nodes[1]
    assert sub.kind == BlockKind.HEADING and "现货" in sub.text
    ch = sub.children[0]
    assert ch.kind == BlockKind.CHAPTER
    assert ch.path.display == "辽宁省现货电能量市场交易实施细则 第一章"
    art = ch.children[0]
    assert art.kind == BlockKind.ARTICLE
    assert art.path.display.endswith("第一章 第一条")


def test_body_lines_not_index_segments():
    """正文行头/人名/表格数字不再进索引路径（text[:12] 病灶）。"""
    pages = [(
        "寻找无边界扩张的行业\n"
        "核心观点\n"
        "陈卓\n"
        "1%\n"
        "未来中国的零售巨头将以电子商务的身份出现。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    disp = _paths(nodes)
    # 标题与栏目标题占段；正文/人名/数据行不占
    segs = [p for _, p in disp if p]
    assert any("寻找无边界" in s for s in segs)
    assert any("核心观点" in s for s in segs)
    joined = " | ".join(segs)
    assert "陈卓" not in joined and "1%" not in joined and "零售巨头" not in joined
    # 正文块 breadcrumb 只到栏目标题（正文行不占段）
    heads = [n for n in nodes if n.kind == BlockKind.HEADING and "核心观点" in n.text]
    body = [n for n in heads[0].children if n.kind == BlockKind.TEXT]
    assert body and body[0].path.display == "核心观点"
    assert body[0].kind == BlockKind.TEXT


def test_article_wrap_line_not_new_para():
    """条款排版折行并回本条（条款即原子块）——曾每折行一个新款+行头占段。"""
    pages = [(
        "第一章总则\n"
        "第一条为规范辽宁电力市场的运营及管理，构建安全、\n"
        "合理、高效的市场体系，制订本规则。\n"
        "第二条本规则适用于辽宁省电力市场。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    art = nodes[0].children[0]
    assert art.kind == BlockKind.ARTICLE
    assert "构建安全、 合理、高效" in art.text or "构建安全、合理" in art.text.replace(" ", "")
    assert not art.children, "折行续行不应生成款节点"


def test_no_heading_false_positive_inside_article():
    """条内折行"按照本规则"不是标题（短行+后缀误判防线）。"""
    pages = [(
        "第一章总则\n"
        "第一条市场成员应当遵纪守法，\n"
        "按照本规则\n"
        "开展交易活动。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    assert [n.kind for n in nodes] == [BlockKind.CHAPTER]
    art = nodes[0].children[0]
    assert art.kind == BlockKind.ARTICLE
    assert "按照本规则" in art.text and "开展交易活动" in art.text
    assert not any(n.kind == BlockKind.HEADING for n in art.walk())


def test_subdoc_title_recognized_mid_document():
    """多分则：上一分则条流之后、页首的分则标题必须识别（曾全漏→同名章撞车）。"""
    pages = [
        ("辽宁省电力市场运营规则\n第一章总则\n第一条为规范运营，制订本规则。\n", 1),
        ("辽宁省现货电能量市场交易实施细则\n第一章总则\n第一条为规范现货交易，制订本细则。\n", 2),
    ]
    nodes, _ = build_ast(pages)
    heads = [n for n in nodes if n.kind == BlockKind.HEADING]
    assert len(heads) == 2, _paths(nodes)
    assert "现货" in heads[1].text
    ch2 = heads[1].children[0]
    assert ch2.path.display == "辽宁省现货电能量市场交易实施细则 第一章"
    # 两个分则的"第一章"路径不撞车
    ch1 = heads[0].children[0]
    assert ch1.path.display != ch2.path.display


def test_heading_seg_space_free():
    """标题段内空白清除（display 以空格连段，段内空格曾把标题劈成碎节点）。"""
    pages = [("辽宁省电力市场运营规则（试行4.0 版征求意见稿）\n", 1)]
    nodes, _ = build_ast(pages)
    assert nodes[0].path.display == "辽宁省电力市场运营规则（试行4.0版征求意见稿）"


def test_glossary_item_lines_merge():
    """名词解释条目排版折行并回同块（曾每行 25 字碎块→检索截断必拒答）。

    现场：p113"4.电力现货市场…：通过交易平台在"与后继行各自成 chunk，
    定义被腰斩；p30"12.统一结算点电价"同样切三段。
    """
    pages = [(
        "附件1\n"
        "名词解释\n"
        "4.电力现货市场（简称“现货市场”）：通过交易平台在\n"
        "日前及更短时间内集中开展的次日、日内至实时调度之前\n"
        "电力交易活动的总称。辽宁现货市场以15 分钟为一个交易\n"
        "出清时段，每个运行日含有96 个交易出清时段。\n"
        "5.日前电能量市场（简称“日前市场”）：运行日提前一\n"
        "天（D-1 日）进行的决定运行日机组组合状态的电能量市场。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    texts = [n.text for r in nodes for n in r.walk() if n.kind == BlockKind.TEXT]
    # 条目 4 三折行并一块；条目 5 另起；各自完整
    assert len(texts) == 2, texts
    assert texts[0].startswith("4.电力现货市场") and "96 个交易出清时段" in texts[0]
    assert "通过交易平台在日前及更短时间内" in texts[0].replace(" ", "")
    assert texts[1].startswith("5.日前电能量市场") and "D-1" in texts[1]


def test_heading_body_closed_line_stays_separate():
    """标题下独立成句的行不并块（研报要点每条一行互不粘连）。"""
    pages = [(
        "核心观点\n"
        "未来中国零售巨头将以电子商务身份出现。\n"
        "线下渠道商转型电商是大势所趋。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    texts = [n.text for r in nodes for n in r.walk() if n.kind == BlockKind.TEXT]
    assert len(texts) == 2, texts


def test_item_wrap_lines_merge():
    """（项）末尾折行并回本项："应提前45个"+"工作日…"曾各自成块（R2 A14 金提示失配根因）。"""
    pages = [(
        "第一章总则\n"
        "第一条售电公司退出市场按以下规定办理。\n"
        "（二）售电公司自愿申请退出售电市场，应提前45个\n"
        "工作日向电力交易机构提交退出申请。\n"
        "（三）公示期满无异议的，方可办理退出手续。\n",
        1,
    )]
    nodes, _ = build_ast(pages)
    items = [n for r in nodes for n in r.walk() if n.kind == BlockKind.ITEM]
    assert len(items) == 2, [n.text for n in items]
    assert "45个工作日" in items[0].text.replace(" ", "")
    assert items[1].text.startswith("公示期满")
