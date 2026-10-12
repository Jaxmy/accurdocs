"""ID 稳定性与幂等测试（计划验收：同一文档重跑两次 ID 一致）。"""

from policyrag.models.ids import ClausePath, make_chunk_id, normalize_text_for_id


def test_normalize_text_for_id_stable():
    # ID 规范化契约：全角转半角 + 彻底去除空白（跨渲染 ID 稳定，计划 §三）
    assert normalize_text_for_id("１２３　ＡＢＣ") == "123ABC"
    assert normalize_text_for_id("  多 个  空白 ") == "多个空白"


def test_make_chunk_id_deterministic():
    path = ClausePath.of("第三章", "第十二条", "第二款")
    a = make_chunk_id("银发〔2024〕15号", "2024-03-01", path, "测试文本")
    b = make_chunk_id("银发〔2024〕15号", "2024-03-01", path, "测试文本")
    assert a == b
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_make_chunk_id_sensitive_to_inputs():
    path = ClausePath.of("第三章", "第十二条")
    base = make_chunk_id("doc1", "v1", path, "文本")
    assert base != make_chunk_id("doc1", "v2", path, "文本")
    assert base != make_chunk_id("doc2", "v1", path, "文本")
    assert base != make_chunk_id("doc1", "v1", ClausePath.of("第四章", "第十二条"), "文本")
    assert base != make_chunk_id("doc1", "v1", path, "文本2")


def test_normalize_affects_id_consistently():
    """全角/半角、空白差异下规范化后 ID 一致（ID 稳定性红线）。"""
    path = ClausePath.of("第十二条")
    a = make_chunk_id("doc1", "v1", path, normalize_text_for_id("１，０００ 万元"))
    b = make_chunk_id("doc1", "v1", path, normalize_text_for_id("1,000万元"))
    assert a == b
