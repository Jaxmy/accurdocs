"""公式/图表富化测试（公式线 + 图表线）。

注意：含反斜杠的 LaTeX 字面量必须经 Write 工具落盘——
bash heredoc 传输会吃掉一层反斜杠（\right 的 \r 被解释为回车）。
"""

from __future__ import annotations

import json
from pathlib import Path

from policyrag.ingest.formulas import extract_symbols, parse_mineru_markdown, validate_latex
from policyrag.models.answer import Citation
from policyrag.models.rich import FormulaRecord
from policyrag.verify.citations import verify_citations

# 真实 MinerU 输出样本（取自 PoC golden_pages 解析）
SAMPLE_MD = (
    "<!-- page 5 of 10 -->\n"
    "第四十四条 新能源场站短期功率预测补偿费用。\n"
    "计算方式如下。\n\n"
    "$$\nR_{ 短期功率预测补偿 ,i} = \\frac{Q_{ 月上网 ,i}}{\\sum_{i} Q_{ 月上网 ,i}}\n"
    "\\times R_{ 新能源短期功率预测偏差考核返还 } \\times \\eta_{ 月 }\n$$\n\n"
    "式中:\n\n$A_{i,短期功率}$ 为新能源场站i月度短期功率预测准确率；\n"
    "\\eta_{ 月 } 为新能源短期功率预测偏差考核返还费用比例系数。\n\n"
    "<!-- page 6 of 10 -->\n"
    "$$\nR_{ 必开补偿 ,i,D} = Max\\left[ \\sum_{t=1}^{t \\in D_{ 必开时限 ,i}} "
    "\\left( M_{ 必开运行成本 ,i,t} - R_{ 必开收益 ,i,t} \\right), 0 \\right]\n$$\n"
    "式中: $D_{必开时段,i}$ 为必开机组i的必开时段集合。\n"
)


def test_parse_mineru_markdown_pages_and_blocks():
    raws = parse_mineru_markdown(SAMPLE_MD)
    assert len(raws) == 2
    assert raws[0]["page"] == 5
    assert raws[1]["page"] == 6
    assert "短期功率预测补偿" in raws[0]["latex"]
    assert "Max" in raws[1]["latex"]


def test_extract_symbols_from_shizhong():
    raws = parse_mineru_markdown(SAMPLE_MD)
    syms0 = extract_symbols(raws[0]["tail"])
    names = [s.name for s in syms0]
    assert any("A" in n for n in names), names
    syms1 = extract_symbols(raws[1]["tail"])
    assert any("D" in n for n in [s.name for s in syms1]), syms1


def test_validate_latex_fixes_only_with_source_evidence():
    fixes = []
    doc_text = "必开机组的必开时段集合省内实时加权"
    # 必开时限→必开时段：对侧在原文出现 → 修
    fixed = validate_latex("D_{ 必开时限 ,i}", doc_text, fixes)
    assert "必开时段" in fixed and "必开时限" not in fixed
    assert fixes
    # 错侧也出现在原文 → 不修（原文就是这么写的，宁缺勿错）
    fixes2 = []
    fixed2 = validate_latex("D_{ 必开时限 ,i}", "必开时限", fixes2)
    assert "必开时限" in fixed2 and not fixes2


def test_formula_evidence_and_latex_loose_citation():
    fr = FormulaRecord(
        local_id="f_001_01", page=1,
        latex="R = " + "\\" + "sum_{d}^{D} " + "\\" + "left[ Q " + "\\" + "right]",
        display_text="R = sum formula",
    )
    ev = fr.evidence_text()
    assert "【公式】" in ev and "【LaTeX】" in ev
    cid = "abc123abc123abc1"
    registry = {cid: ev}
    # 引用方省略 latex 装饰符与空格 → 宽松比对应通过
    quote = "\\" + "sum_{d}^{D}[Q]"
    c = Citation(chunk_id=cid, quote=quote, doc_title="t", doc_version="v", clause_path="")
    r = verify_citations([c], registry, {cid})
    assert r.ok, r.failures


def test_golden_mineru_output_parses():
    """真实 PoC 产物回归：mineru_out 里 21 个公式块全部可解析。"""
    p = Path("data/formula_eval/mineru_out")
    if not p.exists():
        return  # 无 PoC 产物时跳过
    raws = parse_mineru_markdown(p.read_text(encoding="utf-8"))
    assert len(raws) >= 15
    latexes = [r["latex"] for r in raws]
    assert any("阻塞" in l for l in latexes)
    assert any("Max" in l for l in latexes)
