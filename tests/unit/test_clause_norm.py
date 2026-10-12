"""WS-A 条款号归一化测试：跨形态同一化 + 边界防误中。"""

from policyrag.index.tokenizer_zh import tokenize
from policyrag.models.clause_norm import (
    canon_clause_ref,
    clause_alias_token,
    clause_ref_hits,
)
from policyrag.verify.policy import route_risk_tier


class TestCanon:
    def test_forms_map_to_same(self):
        for s in ["第十二条", "第12条", "第１２条", "12条", "Article 12",
                  "Art. 12", "Section 12", "Clause 12", "§12"]:
            assert canon_clause_ref(s) == "条:12", f"{s} → {canon_clause_ref(s)}"

    def test_zhi_suffix(self):
        assert canon_clause_ref("第十二条之二") == "条:12之2"
        assert canon_clause_ref("12条之2") == "条:12之2"

    def test_kinds(self):
        assert canon_clause_ref("第三章") == "章:3"
        assert canon_clause_ref("第五节") == "节:5"
        assert canon_clause_ref("第七款") == "款:7"
        assert canon_clause_ref("（二）") is None   # 项不带"项"字不算条款引用

    def test_cn_large(self):
        assert canon_clause_ref("第一百二十八条") == "条:128"

    def test_non_clause(self):
        assert canon_clause_ref("资本充足率") is None
        assert canon_clause_ref("8%") is None
        assert canon_clause_ref("") is None


class TestHits:
    def test_cross_form(self):
        assert clause_ref_hits("第十二条", "第三章 第十二条 资本充足率……")
        assert clause_ref_hits("12条", "第三章 第十二条 资本充足率……")
        assert clause_ref_hits("第12条", "12条 资本充足率不得低于 8%")
        assert clause_ref_hits("Article 12", "第十二条 资本充足率")

    def test_boundary_no_false_positive(self):
        assert not clause_ref_hits("12条", "第一百一十二条 杠杆率……"), "12条不得命中112条"
        assert not clause_ref_hits("第12条", "第112条 ……")
        assert not clause_ref_hits("第十二条", "第一百一十二条 ……")

    def test_zhi_distinct(self):
        assert not clause_ref_hits("第十二条", "第十二条之一 ……")
        assert clause_ref_hits("第十二条之二", "第十二条之二 ……")

    def test_non_clause_fallback_substring(self):
        assert clause_ref_hits("资本充足率", "关于资本充足率的规定")
        assert not clause_ref_hits("杠杆率", "关于资本充足率的规定")


class TestTokenizerAlias:
    def test_alias_emitted_both_forms(self):
        a = set(tokenize("第十二条"))
        b = set(tokenize("第12条"))
        c = set(tokenize("12条"))
        assert "§条12" in a and "§条12" in b and "§条12" in c, (a, b, c)

    def test_alias_fullwidth(self):
        assert "§条12" in set(tokenize("第１２条"))

    def test_alias_not_for_112(self):
        toks = set(tokenize("第一百一十二条"))
        assert "§条12" not in toks
        assert "§条112" in toks

    def test_raw_tokens_kept(self):
        toks = set(tokenize("第十二条"))
        assert "第十二条" in toks, "原始词元保留（同形态精确匹配不受损）"


class TestRiskTier:
    def test_bare_digit_clause_is_short(self):
        assert route_risk_tier("12条的资本充足率要求是多少？", "quote") == "short"

    def test_fullwidth_clause_is_short(self):
        assert route_risk_tier("第１２条规定了什么？", "quote") == "short"
