"""数值闸机制回归：尾随零归一、引用标记/字母词数值屏蔽、表格必答语境收窄。"""

from policyrag.ingest.normalize import canonical_number, extract_numeric_spans, normalize
from policyrag.verify.coverage import verify_numeric_coverage
from policyrag.verify.deterministic import verify_numbers


class TestCanonicalNumber:
    def test_trailing_zeros(self):
        assert canonical_number("2.40") == "2.4"
        assert canonical_number("40.0") == "40"
        assert canonical_number("12.30") == "12.3"
        assert canonical_number("40.0%") == "40%"

    def test_thousand_sep_and_units(self):
        assert canonical_number("1,000.00 万元") == "1000万元"
        assert canonical_number("20,481") == "20481"


class TestSpanExtraction:
    def _spans(self, text):
        return [s.canonical for s in extract_numeric_spans(normalize(text))]

    def test_alnum_word_not_number(self):
        assert "2" not in self._spans("B2C交易规模占比40.0%")
        assert "40%" in self._spans("B2C交易规模占比40.0%")

    def test_citation_marks_masked(self):
        got = self._spans("见[10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11]和[4581989200ec821c]。")
        assert not got, f"引用标记内数字不得提取，实测 {got}"
        got2 = self._spans("[chunk 4581989200ec821c 【公式】R = 96]。")
        assert got2 == [], f"chunk 包装内数字不得提取，实测 {got2}"

    def test_unit_number_kept(self):
        assert "6" in self._spans("最大充放电功率不小于6MW")
        assert "12" in self._spans("调节容量12MWh")


class TestCoverageScope:
    def test_table_only_question_entities_demanded(self):
        """问 4 国对比：答案给 4 国数值即过——表内他国（231个/120个）不进必答。"""
        q = "中国每百万人拥有大卖场的数量是多少？与法国、美国、韩国相比处于什么水平？"
        ev = ("各国大卖场普及率：中国2.40个，法国25个，美国12.3个，韩国7.60个，"
              "英国231个，德国120个。")
        ans = "中国2.4个/百万人口，法国25个，美国12.3个，韩国7.6个，远低于发达国家。"
        r = verify_numeric_coverage(q, ans, [ev])
        assert r.ok, (r.reason, r.missing_examples)

    def test_precision_form_not_failing(self):
        """2.40（表格）vs 2.4（答案）同值不同形不得误杀（verify_numbers + coverage 双闸）。"""
        q = "普及率是多少？"
        ev = "中国大卖场普及率为2.40个/百万人。"
        ans = "中国大卖场普及率为2.4个/百万人。"
        assert verify_numeric_coverage(q, ans, [ev]).ok
        vr = verify_numbers(ans, [ev])
        assert not vr.failures, vr.failures

    def test_b2c_not_demanded(self):
        """B2C 内嵌数字不得成为数值锚点（曾误判漏答"2"）。"""
        q = "B2C交易规模是多少亿元？占比是多少？"
        ev = "2015年B2C交易规模10,145亿元，占比40.0%。"
        ans = "B2C交易规模10145亿元，占比40%。"
        r = verify_numeric_coverage(q, ans, [ev])
        assert r.ok, (r.reason, r.missing_examples)

    def test_uuid_fragments_not_failing_numbers(self):
        """假 UUID 引用标记碎片不得刷出数值失败（SA01 曾刷数百条）。"""
        ans = ("差价合约电费按96点计算。[10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11]"
               "[f83ee3bc-1a2d-4c6b-8e77-3d5b0a9c2e50]")
        ev = ["差价合约电费按96点结算。"]
        vr = verify_numbers(ans, ev)
        assert not [f for f in vr.failures if "数值" in f], vr.failures[:5]


class TestFracturedDecimal:
    """表格抽取劈开的小数（"1665 .13"）→ 归一为 1665.13（#11 数值误杀根因）。"""

    def test_extract_fractured(self):
        got = [s.canonical for s in extract_numeric_spans(normalize("规模 1665 .13 亿美元"))]
        assert "1665.13" in got, got

    def test_verify_answer_clean_vs_fractured_evidence(self):
        ans = "2010年电子商务规模为1665.13亿美元。"
        ev = ["2010年美国零售业，电子商务规模 1665 .13 亿美元"]
        vr = verify_numbers(ans, ev)
        assert not [f for f in vr.failures if "数值" in f], vr.failures

    def test_coverage_fractured(self):
        q = "电子商务规模是多少亿美元？"
        ev = "2010年规模比较：38877.25 | 1665 .13 | 2600 | 187"
        ans = "电子商务规模1665.13亿美元。"
        r = verify_numeric_coverage(q, ans, [ev])
        assert r.ok, (r.reason, r.missing_examples)
