"""中文条款分块边界测试（计划 §五.1 红线）。"""

from policyrag.ingest.chunker import ChunkConfig, chunk_document
from policyrag.ingest.structure import build_ast
from policyrag.models.document import DocAST, TableBlock


DOC = """第一章 总则

第一条 为加强贷款管理，防范金融风险，根据《中华人民共和国银行业监督管理法》等法律法规，制定本办法。

第二条 本办法所称贷款，是指贷款人向借款人提供的并按约定的利率和期限还本付息的货币资金。

（一）短期贷款，是指贷款期限在一年以内（含一年）的贷款；
（二）中长期贷款，是指贷款期限在一年以上的贷款。

第三条 贷款人应当建立审慎的贷款管理制度。资本充足率不得低于 8%。
但是，经国务院银行业监督管理机构批准的除外。

第二章 发放

第四条 贷款人发放贷款，借款人应当提供担保。贷款额度不得超过 1,000 万元。
"""


def _ast():
    nodes, _ = build_ast([(DOC, 1)], tables=[])
    return DocAST(doc_id="测试办法", version="2024-01-01", title="测试办法", nodes=nodes)


def test_article_boundaries_respected():
    ast = _ast()
    chunks = chunk_document(ast)
    texts = [c.text for c in chunks]
    # 每条独立成块
    assert any("为加强贷款管理" in t for t in texts)
    assert any("本办法所称贷款" in t for t in texts)
    # 条款号进入 path
    paths = [c.path.key for c in chunks]
    assert any("第一条" in p for p in paths)
    assert any("第二条" in p for p in paths)


def test_items_under_article():
    ast = _ast()
    chunks = chunk_document(ast)
    item_chunks = [c for c in chunks if "（一）" in c.path.key or "（一）" in c.text]
    assert item_chunks, "枚举项应被识别"
    assert any("短期贷款" in c.text for c in item_chunks)


def test_but_clause_stays_with_article():
    """但书句必须与本款同块，不被切断（计划 §五.1）。"""
    ast = _ast()
    chunks = chunk_document(ast)
    art3 = [c for c in chunks if "第三条" in c.path.key]
    assert art3
    joined = " ".join(c.text for c in art3)
    assert "但是" in joined or "但书" in joined or "除外" in joined
    # 且与 8% 在同一条块内
    assert any("8%" in c.text and ("除外" in c.text or "但是" in c.text) for c in art3)


def test_long_article_splits_with_parent():
    long_text = "第一条 " + "本条内容规定了具体事项。" * 60  # 远超 max_chars
    nodes, _ = build_ast([(long_text, 1)], tables=[])
    ast = DocAST(doc_id="长文", version="v1", title="长文", nodes=nodes)
    chunks = chunk_document(ast, ChunkConfig(max_chars=200, target_chars=150, min_chars=20))
    assert len(chunks) > 1
    # 子块带 parent 引用与"续"路径
    assert any(c.parent_chunk_id or "续" in c.path.key for c in chunks[1:])


def test_table_row_chunks():
    tb = TableBlock(caption="表1 贷款分类", headers=["类型", "利率"],
                    rows=[["短期", "5%"], ["长期", "6%"]])
    nodes, _ = build_ast([("第一条 贷款分类如下。", 1)], tables=[tb])
    ast = DocAST(doc_id="表测", version="v1", title="表测", nodes=nodes)
    chunks = chunk_document(ast)
    assert any(c.row_index is not None for c in chunks)
    row_chunk = next(c for c in chunks if c.row_index == 0)
    assert "5%" in row_chunk.evidence_text()
    assert "利率" in row_chunk.evidence_text()  # 表头上下文随行携带


def test_id_stable_across_runs():
    ast1 = _ast()
    ast2 = _ast()
    ids1 = [c.chunk_id for c in chunk_document(ast1)]
    ids2 = [c.chunk_id for c in chunk_document(ast2)]
    assert ids1 == ids2
