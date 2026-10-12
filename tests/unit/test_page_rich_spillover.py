"""公式页溢出补位回归：条款页尾"计算公式如下："→ 续页公式块进证据/附图。

实测 bug：发电侧省内实时公式 f_227_01 在 p227（转写完好、高置信），
第二十八条正文在 p226——检索不中（纯 LaTeX 文本）、家族扩展够不着
（"公式/"独立树）、同页扩展查不到续页 → 三头漏。修复：符号表进文本 +
页邻域（±1）补位 + 公式锚定向扩展。
"""

from policyrag.ingest.enrichment import merge_symbol_lines
from policyrag.models.retrieval import Evidence
from policyrag.query.pipeline import (
    PipelineDeps, QueryPipeline, dedupe_rich_lists, neighborhood_pages,
)


class _FakeLLM:
    name = "fake"
    caps = type("C", (), {"structured_output": False, "json_mode": True})()

    def complete(self, *a, **k):
        raise AssertionError("不应调用 LLM")


def _pipe(live_page_rich):
    deps = PipelineDeps(index=None, llm=_FakeLLM(), registry={},
                        meta_lookup={}, corpus_version="t",
                        live_page_rich=live_page_rich)
    return QueryPipeline(deps, trace_root="tests/.tmp/spill_traces")


def test_neighborhood_pages():
    assert neighborhood_pages(226) == [225, 226, 227]
    assert neighborhood_pages(0) == [0]


def test_merge_symbol_lines():
    syms = [{"name": "R_{省内实时,i}", "desc": "发电企业i月度省内实时市场电能量电费"}]
    out = merge_symbol_lines("【公式】R = \\sum", syms)
    assert "【符号】R_{省内实时,i} 为 发电企业i月度省内实时市场电能量电费" in out
    # 幂等：已有【符号】不重复追加
    assert merge_symbol_lines(out, syms) == out
    assert merge_symbol_lines("【公式】R", []) == "【公式】R"


def test_anchor_block_pulls_next_page_formula():
    """条款块含"计算公式如下"锚 → 续页（+1）公式块补进证据；无关公式不进。"""
    kids = [
        {"chunk_id": "f_227_01",
         "text": "【公式】R_{省内实时,i} = \\sum (Q_{日上网,i,t} \\times LMP_{省内实时,i,t})\n"
                 "【符号】R_{省内实时,i} 为 发电企业i月度省内实时市场电能量电费",
         "breadcrumb": "第227页 公式f_227_01", "page": 227},
        {"chunk_id": "f_227_03",
         "text": "【公式】R_{发电侧价差调整} = \\sum (Q_{日上网,t} \\times (LMP' - LMP))",
         "breadcrumb": "第227页 公式f_227_03", "page": 227},
    ]

    def live_page_rich(doc_key, page):
        return kids if page == 227 else []

    p = _pipe(live_page_rich)
    p.deps.meta_lookup = {"para28": {"doc_key": "LIAO@v1"}}
    evs = [Evidence(chunk_id="para28",
                    text="省内实时市场上网电量与省内实时市场节点电价或统一结算点电价"
                         "计算省内实时市场电能电费，计算公式如下：",
                    doc_title="LIAO", doc_version="v1",
                    clause_path="第四章 第一节 第二十八条", page=226)]
    out = p._expand_page_rich("省内实时市场电能量电费怎么计算？", evs)
    ids = [e.chunk_id for e in out]
    assert "f_227_01" in ids, "续页相关公式应补进证据"
    assert "f_227_03" not in ids, "同页无关公式（价差调整）不该混入"
    assert out[-1].page == 227


def test_no_anchor_no_pull():
    p = _pipe(lambda dk, pg: [{"chunk_id": "fx",
                               "text": "【公式】R_{省内实时} = 1",
                               "breadcrumb": "第9页 公式f_009_01", "page": 9}])
    p.deps.meta_lookup = {"para": {"doc_key": "LIAO@v1"}}
    evs = [Evidence(chunk_id="para", text="市场成员包括发电企业、售电公司。",
                    doc_title="T", doc_version="v1", clause_path="第二条", page=8)]
    out = p._expand_page_rich("省内实时市场电能量电费怎么计算？", evs)
    assert [e.chunk_id for e in out] == ["para"]


def test_dedupe_rich_lists_same_latex_and_image():
    """同式异块/同图异块只留一个（重复引用的附图和公式只留一个）。"""
    fs = [
        {"chunk_id": "a", "latex": "R = " + "\\" + "sum_{d} ( Q )", "image": "x/f_1.png"},
        {"chunk_id": "b", "latex": "R = " + "\\" + "sum_{d}(Q)", "image": "y/f_2.png"},
        {"chunk_id": "c", "latex": "R2 = Q", "image": "z/f_3.png"},
    ]
    out_f, _ = dedupe_rich_lists(fs, [])
    assert [f["chunk_id"] for f in out_f] == ["a", "c"]
    cs = [{"chunk_id": "c1", "image": "p/a.png", "title": "t1"},
          {"chunk_id": "c2", "image": "q/a.png", "title": "t2"},
          {"chunk_id": "c3", "image": "q/b.png", "title": "t3"}]
    _, out_c = dedupe_rich_lists([], cs)
    assert [c["chunk_id"] for c in out_c] == ["c1", "c3"]
