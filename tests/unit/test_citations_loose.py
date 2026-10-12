"""引用校验符号宽松层：Symbol 乱码公式被模型丢弃不可见字符仍须可溯源。"""

from policyrag.models.answer import Citation
from policyrag.verify.citations import verify_citations

CHUNK_ID = "7847d8eae46cfa27"
# 真实场景：215 页正文内嵌 Symbol 字体乱码公式（≥× 为 PUA）
CHUNK = ("燃煤基准电价的1.5 倍时，在结算环节将市场各时段节电点价等比例缩小。"
         "计算公式如下：若 1.5 P P ≥× 燃煤基准电价 实时全天统一结算点加权均价")


def _cite(quote):
    return Citation(chunk_id=CHUNK_ID, quote=quote, doc_title="", doc_version="",
                    clause_path="第十三条")


def test_symbol_dropped_quote_passes_loose_tier():
    """模型引用时丢掉 PUA/符号字符（实测形态）→ 符号宽松层放行（可溯源）。"""
    r = verify_citations([_cite("若 1.5 P P  燃煤基准电价 实时全天统一结算点加权均价")],
                         {CHUNK_ID: CHUNK}, {CHUNK_ID})
    assert r.ok, r.failures


def test_exact_quote_still_passes():
    r = verify_citations([_cite("在结算环节将市场各时段节电点价等比例缩小")],
                         {CHUNK_ID: CHUNK}, {CHUNK_ID})
    assert r.ok


def test_fabricated_content_still_fails():
    """宽松层只放行符号差异——改数字/改词仍然拦截。"""
    r = verify_citations([_cite("若 1.8 P P 燃煤基准电价 实时全天统一结算点加权均价")],
                         {CHUNK_ID: CHUNK}, {CHUNK_ID})
    assert not r.ok
    r2 = verify_citations([_cite("若 1.5 P P 虚构内容完全不存在")],
                          {CHUNK_ID: CHUNK}, {CHUNK_ID})
    assert not r2.ok
