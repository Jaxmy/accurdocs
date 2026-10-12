"""breadcrumb 清洗回归（#10）：句中碎片段剔除、结构段/干净标题保留。"""

from policyrag.models.clause_norm import clean_breadcrumb


def test_polluted_segments_dropped():
    # 数学模型附件实测污染形态
    out = clean_breadcrumb("第七章 第四节 第八十条 经辽宁省工业和信息化厅、 （一）")
    assert out == "第七章 第四节 第八十条 （一）", out
    # 注册细则续块实测污染形态
    out2 = clean_breadcrumb("第三章 第七条 得、按规定时限正在办理或 （二）")
    assert out2 == "第三章 第七条 （二）", out2


def test_structural_and_titles_kept():
    assert clean_breadcrumb("第三章 第七条 （一）") == "第三章 第七条 （一）"
    assert clean_breadcrumb("第三章 注册基本条件") == "第三章 注册基本条件"
    assert clean_breadcrumb("第十二条 续1") == "第十二条 续1"
    assert clean_breadcrumb("第225页 公式 f_225_01") == "第225页 公式 f_225_01"
    assert clean_breadcrumb("第七章 （第 82 页表） 表 行1") == "第七章 表 行1"


def test_long_fragment_dropped():
    out = clean_breadcrumb("第十三条 燃煤基准电价的1.5 倍")
    assert out == "第十三条", out


def test_empty_and_none():
    assert clean_breadcrumb("") == ""
    assert clean_breadcrumb(None) == ""
