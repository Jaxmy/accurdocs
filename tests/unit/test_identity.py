"""WS-E 文档身份治理测试：标题提取链 / 内容指纹 / 近似版本 / 防覆盖。"""

import os
import uuid

from policyrag.ingest.identity import (
    clean_filename_stem,
    content_fingerprint,
    decide_identity,
    extract_title,
    hamming,
    resolve_doc_key,
    simhash64,
    title_similar,
)
from policyrag.index.stores.sqlite_meta import MetaStore


class TestTitleChain:
    def test_book_title(self):
        title, src = extract_title("封面\n《辽宁省电力市场运营基本规则》\n第一章 总则", "hash_x")
        assert title == "辽宁省电力市场运营基本规则" and src == "book_title"

    def test_type_suffix_line(self):
        text = "国家能源局\n电力市场运营基本规则（试行）\n第一章 总则\n第一条 为规范……"
        title, src = extract_title(text, "x")
        assert title == "电力市场运营基本规则（试行）" and src == "type_suffix"

    def test_garbage_page_number_rejected(self):
        text = "— 93 —\n2025 年月\n辽宁省电力市场运营基本规则及配套实施细则\n第一章"
        title, src = extract_title(text, "x")
        assert "93" not in title and "年月" not in title
        assert "辽宁省电力市场" in title

    def test_filename_fallback(self):
        title, src = extract_title("！！！\n@@@\n###\n####\n%%%%\n!!!!", "514fd6d7f0_测试贷款办法")
        assert src == "filename"
        assert "测试贷款办法" in title

    def test_watermark_rejected(self):
        text = "仅供内部学习使用\n下载更多资料\n绿色金融指引通知\n第一章"
        title, _ = extract_title(text, "x")
        assert "仅供" not in title and "下载" not in title

    def test_clean_filename_stem(self):
        assert "办法" in clean_filename_stem("514fd6d7f0_测试办法")
        assert "办法" in clean_filename_stem("测试办法-副本(2)")
        assert "办法" in clean_filename_stem("测试办法_副本")


class TestFingerprint:
    def test_stable(self):
        assert content_fingerprint("同一文本") == content_fingerprint("同一文本")
        assert content_fingerprint("同一文本") != content_fingerprint("不同文本")

    def test_simhash_near(self):
        a = "辽宁省电力市场运营基本规则及配套实施细则试行版全文共计若干章节条款"
        b = "辽宁省电力市场运营基本规则及配套实施细则试行版全文共计若干章节条款。"
        assert hamming(simhash64(a), simhash64(b)) <= 6

    def test_simhash_far(self):
        a = "辽宁省电力市场运营基本规则及配套实施细则试行版全文共计若干章节条款"
        c = "商业银行资本管理办法完全不同的文档内容用于测试远距离相似度表现"
        assert hamming(simhash64(a), simhash64(c)) > 6

    def test_title_similar(self):
        assert title_similar("《辽宁省电力市场规则》", "辽宁省电力市场规则") > 0.9
        assert title_similar("绿色金融指引", "商业银行资本管理办法") < 0.5


class TestIdentityAndVersions:
    def _rows(self, rows):
        return rows

    def test_new_doc_new_group(self):
        ident = decide_identity(doc_id="甲办法", version="2025-01-01", title="甲办法",
                                doc_no="", raw_text="甲办法正文" * 50, existing=[])
        assert ident.doc_group == "甲办法" and ident.version_seq == 1 and ident.is_latest

    def test_near_dup_joins_group(self):
        t = "辽宁省电力市场运营基本规则及配套实施细则" + "正文内容" * 80
        existing = [{
            "doc_key": "辽宁规则@undated", "doc_id": "辽宁规则", "version": "undated",
            "title": "辽宁省电力市场运营基本规则及配套实施细则", "doc_group": "辽宁规则",
            "version_seq": 1, "content_hash": content_fingerprint(t),
            "simhash": str(simhash64(t[:20000])), "meta_json": "{}",
        }]
        ident = decide_identity(doc_id="辽宁规则副本", version="undated",
                                title="辽宁省电力市场运营基本规则及配套实施细则（试行）",
                                doc_no="", raw_text=t + "修订补充若干字", existing=existing)
        assert ident.doc_group == "辽宁规则"
        assert ident.version_seq == 2
        assert ident.near_dup_of == "辽宁规则@undated"

    def test_unrelated_not_grouped(self):
        existing = [{
            "doc_key": "甲@v1", "doc_id": "甲", "version": "v1",
            "title": "商业银行资本管理办法", "doc_group": "甲",
            "version_seq": 1, "content_hash": "abc",
            "simhash": str(simhash64("商业银行资本管理办法核心一级资本充足率要求" * 5)),
            "meta_json": "{}",
        }]
        ident = decide_identity(doc_id="乙", version="undated",
                                title="绿色金融指引通知", doc_no="",
                                raw_text="绿色信贷环境信息披露要求与碳减排支持工具" * 30,
                                existing=existing)
        assert ident.doc_group == "乙", "不同文档不得因宽松规则误并组"

    def test_same_title_different_family_not_grouped(self):
        """《通知》类泛标题同名不许单靠标题并组。"""
        existing = [{
            "doc_key": "甲@v1", "doc_id": "甲", "version": "v1",
            "title": "关于加强电力市场管理的通知", "doc_group": "甲",
            "version_seq": 1, "content_hash": "x",
            "simhash": str(simhash64("电力市场管理第一份通知完全不同的具体条款内容" * 10)),
            "meta_json": "{}",
        }]
        ident = decide_identity(doc_id="乙", version="undated",
                                title="关于加强电力市场管理的通知", doc_no="",
                                raw_text="另一份内容迥异的通知文本讲的是燃煤发电价格监管事项" * 10,
                                existing=existing)
        assert ident.doc_group == "乙"

    def test_resolve_doc_key_collision_and_noop(self):
        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        m = MetaStore(f"{tmp}/meta.sqlite3")
        m.upsert_document("甲", "2025-01-01", "甲办法", content_hash="hashA")
        # 同 key 异指纹 → -r2
        v, noop = resolve_doc_key(m, "甲", "2025-01-01", "hashB")
        assert v == "2025-01-01-r2" and not noop
        # 同指纹 → noop
        v2, noop2 = resolve_doc_key(m, "甲", "2025-01-01", "hashA")
        assert noop2 is True

    def test_noop_short_circuit_via_ingest(self):
        from pathlib import Path
        from policyrag.ingest.pipeline import ingest_file
        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp, exist_ok=True)
        p = Path(tmp) / "doc.md"
        p.write_text("# 测试办法通知\n\n第一条 资本充足率不得低于 8%。\n", encoding="utf-8")
        m = MetaStore(f"{tmp}/meta.sqlite3")
        r1 = ingest_file(p, m)
        assert r1["status"] == "ok"
        r2 = ingest_file(p, m)
        assert r2["status"] == "noop", f"同内容重传应 noop，实测 {r2['status']}"
        assert r2["doc_key"] == r1["doc_key"]
