"""展示层条款拼合回归：整条拼合、富化块不拼、先拼后分页、block_index、chunk_id 稳定红线。"""

from policyrag.service import reader_view
from policyrag.service.reader_view import (
    block_index_of,
    chunk_to_primary,
    display_blocks,
    merge_key,
)


def _row(cid, path_key, breadcrumb=None, kind="text", page=1, text=None):
    return {"chunk_id": cid, "path_key": path_key,
            "breadcrumb": breadcrumb if breadcrumb is not None else path_key,
            "kind": kind, "page": page,
            "text": text if text is not None else f"正文{cid}",
            "meta_json": "{}"}


class TestMergeKey:
    def test_clause_truncates_at_article(self):
        assert merge_key(_row("a", "第三章/第十二条")) == "第三章/第十二条"
        assert merge_key(_row("b", "第三章/第十二条/续1")) == "第三章/第十二条"
        assert merge_key(_row("c", "第三章/第十二条/第一款")) == "第三章/第十二条"

    def test_no_article_falls_back_to_cont_strip(self):
        assert merge_key(_row("a", "第三章/概述")) == "第三章/概述"
        assert merge_key(_row("b", "第三章/概述/续1")) == "第三章/概述"

    def test_rich_and_table_never_merge(self):
        assert merge_key(_row("f", "公式/f_001_01", kind="formula")) is None
        assert merge_key(_row("c", "图表/g_001_01", kind="chart")) is None
        assert merge_key(_row("t", "第十二条/表", kind="table")) is None
        assert merge_key(_row("r", "第十二条/表/行1", kind="table")) is None
        assert merge_key(_row("x", "")) is None

    def test_fullwidth_and_cn_numerals(self):
        assert merge_key(_row("a", "第１２条/续1")) == "第１２条"
        assert merge_key(_row("b", "第十二条之三/第一款")) == "第十二条之三"


class TestDisplayBlocks:
    def test_continuation_pieces_merge(self):
        """一条的续块拼成一块：chunk_ids 有序、主 id=基块、text 换行连接。"""
        rows = [
            _row("base", "第三章/第十二条", text="开头"),
            _row("xu1", "第三章/第十二条/续1", text="中段"),
            _row("xu2", "第三章/第十二条/续2", text="结尾"),
        ]
        blocks = display_blocks(rows)
        assert len(blocks) == 1
        b = blocks[0]
        assert b["chunk_id"] == "base"
        assert b["chunk_ids"] == ["base", "xu1", "xu2"]
        assert b["text"] == "开头\n中段\n结尾"
        assert [p["chunk_id"] for p in b["pieces"]] == ["base", "xu1", "xu2"]

    def test_clause_mode_absorbs_kuan_xiang(self):
        rows = [
            _row("base", "第十二条", text="条文"),
            _row("k1", "第十二条/第一款", text="款一"),
            _row("x1", "第十二条/（一）", text="项一"),
        ]
        blocks = display_blocks(rows)
        assert len(blocks) == 1
        assert blocks[0]["chunk_ids"] == ["base", "k1", "x1"]

    def test_cont_mode_keeps_kuan_solo(self, monkeypatch):
        monkeypatch.setattr(reader_view, "MERGE_MODE", "cont")
        rows = [
            _row("base", "第十二条", text="条文"),
            _row("xu1", "第十二条/续1", text="续段"),
            _row("k1", "第十二条/第一款", text="款一"),
        ]
        blocks = display_blocks(rows)
        ids = [b["chunk_id"] for b in blocks]
        assert ids == ["base", "k1"]          # 款独立成块
        assert blocks[0]["chunk_ids"] == ["base", "xu1"]   # 续块仍拼

    def test_kuan_sort_interleaving_still_merges(self):
        """款(U+6B3E) < 续(U+7EED)：非连续段也要拼进同一条。"""
        rows = [
            _row("base", "第十二条"),
            _row("k1", "第十二条/第一款"),
            _row("xu1", "第十二条/续1"),
        ]
        blocks = display_blocks(rows)
        assert len(blocks) == 1
        assert blocks[0]["chunk_ids"] == ["base", "k1", "xu1"]

    def test_formula_chart_table_stay_solo(self):
        rows = [
            _row("t1", "第十二条", text="条文"),
            _row("fm", "公式/f_001_01", kind="formula", page=1),
            _row("ch", "图表/g_001_01", kind="chart", page=1),
            _row("tr", "第十二条/表", kind="table"),
            _row("r1", "第十二条/表/行1", kind="table"),
            _row("xu1", "第十二条/续1", text="续段"),
        ]
        blocks = display_blocks(rows)
        by_id = {b["chunk_id"]: b for b in blocks}
        assert by_id["fm"]["chunk_ids"] == ["fm"]
        assert by_id["ch"]["chunk_ids"] == ["ch"]
        assert by_id["tr"]["chunk_ids"] == ["tr"]
        assert by_id["r1"]["chunk_ids"] == ["r1"]
        # 文本条款照拼，不被夹在中间的富化块打断
        assert by_id["t1"]["chunk_ids"] == ["t1", "xu1"]

    def test_empty_path_key_solo(self):
        rows = [_row("a", ""), _row("b", "第十二条"), _row("c", "第十二条/续1")]
        blocks = display_blocks(rows)
        assert [b["chunk_ids"] for b in blocks] == [["a"], ["b", "c"]]

    def test_merge_then_slice_clause_never_straddles(self):
        rows = [_row("base", "第十二条", page=1)] + [
            _row(f"xu{i}", f"第十二条/续{i}", page=1) for i in range(1, 4)]
        rows += [_row("next", "第十三条", page=2)]
        blocks = display_blocks(rows)
        assert len(blocks) == 2            # 展示块计数：拼合后 2 块，非 5 行
        page = blocks[0:1]                 # 模拟 limit=1 分页
        assert page[0]["chunk_ids"] == ["base", "xu1", "xu2", "xu3"]

    def test_redline_chunk_id_stable(self):
        """红线：展示块 chunk_ids 并集 == 输入集合，无丢失、无再生成。"""
        rows = [
            _row("base", "第十二条"), _row("xu1", "第十二条/续1"),
            _row("k1", "第十二条/第一款"),
            _row("fm", "公式/f1", kind="formula"),
            _row("solo", "前言"),
            _row("orphan", ""),
        ]
        blocks = display_blocks(rows)
        union = {cid for b in blocks for cid in b["chunk_ids"]}
        assert union == {r["chunk_id"] for r in rows}


class TestBlockIndex:
    def test_members_share_block_index(self):
        rows = [
            _row("base", "第十二条", page=1), _row("xu1", "第十二条/续1", page=1),
            _row("xu2", "第十二条/续2", page=1), _row("next", "第十三条", page=2),
        ]
        assert block_index_of(rows, "base") == 0
        assert block_index_of(rows, "xu1") == 0
        assert block_index_of(rows, "xu2") == 0
        assert block_index_of(rows, "next") == 1
        assert block_index_of(rows, "missing") is None

    def test_rich_block_own_index(self):
        rows = [_row("t1", "第十二条", page=1), _row("fm", "公式/f1", kind="formula", page=2)]
        assert block_index_of(rows, "fm") == 1

    def test_chunk_to_primary(self):
        rows = [_row("base", "第十二条"), _row("xu1", "第十二条/续1")]
        assert chunk_to_primary(rows) == {"base": "base", "xu1": "base"}
