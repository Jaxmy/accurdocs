"""测试战役判分：gui_events/ + mcp_out/ → grades.json + 终端摘要。

金提示（qid → {status, answer_re, cite_pages, formula_re, synth_re}）+ 质量红线检查。
判定 OK / BUG / REFUSED_OK。
"""
import json

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
import re
from pathlib import Path

ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10"

# 金提示：status=answered 时 answer 正则（fold 去空白后）；cite_pages=应出现页码；
# refuse=True 时期望拒答。
GOLD = {
    "S01": {"ans": r"合同容量\s*[×xX*]\s*24", "pages": [41], "note": "净合约量公式"},
    "S02": {"ans": r"6\s*(MW|兆瓦)", "ans2": r"12\s*(MWh|兆瓦时)", "pages": [202], "note": "储能准入 6MW/12MWh"},
    "S03": {"ans": r"(阻塞|R_?\{?\s*阻塞)", "pages": [239], "formula": r"(sum|∑)", "note": "阻塞三重∑"},
    "S04": {"ans": r"(负荷平衡|出力之和|系统负荷)", "pages": list(range(135, 181)), "note": "SCUC 负荷平衡约束"},
    "S05": {"ans": r"(补全|拟合|插值)", "pages": [308], "note": "数据补全"},
    "S06": {"ans": r"(套餐)", "pages": [316], "note": "零售套餐类型"},
    "S07": {"ans": r"2\.4", "ans2": r"(25|12\.3|7\.6)", "pages": [11], "note": "大卖场普及率表"},
    "S08": {"refuse": True, "note": "库外阶梯电价必拒"},
    "M01": {"ans": r"(价差调整|LMP)", "pages": [227], "formula": r"(sum|∑)", "note": "价差调整公式"},
    "M02": {"ans": r"典型曲线", "ans2": r"分时段", "pages": [34, 35, 36, 46, 47, 14, 15, 33, 51], "note": "直接交易两类型"},
    "M03": {"ans": r"38877|1665\.13|2600|187", "pages": [55], "note": "2010 美国零售规模对比"},
    "M04": {"ans": r"(电能量电费|五)", "pages": [231], "formula": r"(C_?\{?\s*电能|分项)", "note": "C_电能五分项"},
    "M05": {"ans": r"5\s*%", "pages": [74, 75, 76, 77, 78, 79, 73, 80], "note": "报价段长度 5%"},
    "M06": {"ans": r"20,?481|20481", "ans2": r"10,?145|40\.0", "pages": [32], "note": "网购 2015E 中性"},
    "M07": {"ans": r"(分摊)", "pages": [197], "formula": r"(sum|∑|frac)", "note": "调频费用分摊"},
    "M08": {"ans": r"(功率预测|预测偏差|补偿)", "pages": [238], "formula": r"(sum|∑|frac)", "note": "功率预测补偿分式"},
    "SA01": {"ans": r"(省间|省内)", "formula": r"(sum|∑)", "pages": [225, 226], "synth": True, "note": "差价合约双∑+分析"},
    "SA02": {"ans": r"(分摊|分子|分母)", "pages": [236, 237], "formula": r"(frac|sum|∑)", "synth": True, "note": "补偿分摊分式+分析"},
    "SA03": {"ans": r"(必开|Max|max)", "pages": [96, 97, 98, 99, 100], "synth": True, "note": "必开 Max[∑,0] 对比分析"},
    "SA04": {"ans": r"(备用|爬坡)", "pages": list(range(135, 181)), "synth": True, "note": "备用/爬坡约束+分析"},
    "SA05": {"ans": r"50", "ans2": r"33", "pages": [18], "synth": True, "note": "敏感性表 50→33"},
    "SA06": {"ans": r"(1\s*美金|末端|53)", "pages": [15, 16, 17], "synth": True, "note": "物流成本长尾分析"},
    "SA07": {"ans": r"(虚拟电厂)", "pages": [256], "formula": r"(sum|∑|frac|5)", "synth": True, "note": "虚拟电厂五式+分析"},
    "SA08": {"refuse": True, "note": "库外碳价走势+投资建议必拒"},
    "MA01": {"ans": r"(总电费)", "pages": [251], "synth": True, "note": "发电总电费科目"},
    "MA02": {"ans": r"(售电公司)", "pages": [253], "synth": True, "note": "售电公司总电费+差异"},
    "MA03": {"ans": r"(B2C|综合模式|平台)", "pages": [2, 3, 127], "synth": True, "note": "B2C 综合论据"},
    "MA04": {"ans": r"(注册|基本条件|准入)", "pages": list(range(268, 299)), "synth": True, "note": "注册条件+分析"},
    "MA05": {"ans": r"(京东|苏宁)", "pages": [104, 111, 112, 113], "synth": True, "note": "仓储布局对比"},
    "MA06": {"ans": r"(里程|kp|性能)", "pages": [193, 194], "synth": True, "note": "调频里程价格+kp"},
    "MA07": {"ans": r"(偏差|返还)", "pages": [242, 243, 247], "formula": r"(sum|∑)", "synth": True, "note": "省间偏差+返还"},
    "MA08": {"ans": r"(缺额|考核)", "ans2": r"(补全|拟合)", "pages": [248, 308], "synth": True, "note": "缺额考核+补全"},
}

ID_RE = re.compile(r"\[[0-9a-fA-F]{16}\]")


def fold(s):
    return re.sub(r"\s+", "", s or "")


def grade_env(qid, env, source):
    """单个 envelope 判分 → (verdict, issues[])。"""
    g = GOLD.get(qid, {})
    issues = []
    st = env.get("status")
    if g.get("refuse"):
        if st == "refused":
            return "REFUSED_OK", issues
        issues.append(f"应拒答实际 {st}")
        return "BUG", issues
    if st != "answered":
        issues.append(f"未回答 status={st} refusal={(env.get('refusal') or {}).get('detail', '')[:60]}")
        return "BUG", issues
    ans = env.get("answer") or ""
    f = fold(ans)
    if g.get("ans") and not re.search(g["ans"], f):
        issues.append(f"金提示未命中: {g['ans']} ({g.get('note','')})")
    if g.get("ans2") and not re.search(g["ans2"], f):
        issues.append(f"金提示2未命中: {g['ans2']}")
    # 引用页码
    pages = set()
    for c in env.get("citations") or []:
        d = c.get("display") or c
        if d.get("page"):
            pages.add(d["page"])
    if g.get("pages") and not (pages & set(g["pages"])):
        issues.append(f"引用页码 {sorted(pages)} 不含期望 {g['pages'][:6]}")
    # 公式
    if g.get("formula"):
        fms = env.get("formulas") or []
        latex = " ".join(fold((x.get("latex") or "")) for x in fms)
        if fms and not re.search(g["formula"], latex):
            issues.append(f"公式结构未命中 {g['formula']}")
    # 质量红线
    if ID_RE.search(ans):
        issues.append("答案残留原子 ID")
    if "转写进行中" in ans:
        issues.append("答案复述'转写进行中'占位文案")
    for fl in (env.get("verification") or {}).get("failures") or []:
        if "已删除" in fl:
            issues.append(f"校验删句: {fl[:50]}")
    if g.get("synth") and source == "gui":
        pass  # synthesis 检查在事件层
    return ("OK" if not issues else "BUG"), issues


def grade_gui():
    rows = []
    for p in sorted((ROOT / "gui_events").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        iid = d["input_id"]
        if d.get("plan_error") or d.get("exec_error"):
            rows.append({"qid": iid, "src": "gui", "verdict": "BUG",
                         "issues": [d.get("plan_error") or d.get("exec_error")]})
            continue
        # 拆分合理性按题干数重判（旧记录里 plan_split_ok 按问号计数，批量题误报）
        qs_plan = ((d.get("plan") or {}).get("plan") or {}).get("items") or []
        qtext = d.get("query") or ""
        numbered = len(re.findall(r"(?m)^\s*\d+[.、]", qtext))
        qmarks = numbered if numbered else qtext.count("？") + qtext.count("?")
        if qs_plan and len(qs_plan) < max(1, qmarks):
            rows.append({"qid": iid, "src": "gui", "verdict": "BUG",
                         "issues": ["拆分数量不符"]})
        qmap = {"MB1": ["M01", "M02", "M03"], "MB2": ["M04", "M05", "M06"],
                "MB3": ["M07", "M08"], "MAB1": ["MA01", "MA02", "MA03"],
                "MAB2": ["MA04", "MA05", "MA06"], "MAB3": ["MA07", "MA08"]}
        qids = qmap.get(iid, [iid])
        qdones = [evd["ev"] for evd in d.get("events", [])
                  if evd["ev"].get("type") == "q_done"]
        if iid not in qmap and len(qdones) > 1:
            # 单输入被拆多子答：金提示（ans/ans2）可能分落不同子答——按合并答案判分
            merged = {"status": "answered", "answer": "", "citations": [],
                      "formulas": [], "verification": {}}
            for ev in qdones:
                env = ev.get("envelope") or {}
                merged["answer"] += "\n" + (env.get("answer") or "")
                merged["citations"] += env.get("citations") or []
                merged["formulas"] += env.get("formulas") or []
                merged["status"] = env.get("status") or merged["status"]
            verdict, issues = grade_env(iid, merged, "gui")
            rows.append({"qid": iid, "src": "gui", "verdict": verdict,
                         "issues": issues, "status": merged["status"],
                         "trace": qdones[-1].get("envelope", {}).get("trace_id"),
                         "ans_head": merged["answer"][:70]})
        else:
            i = 0
            for evd in d.get("events", []):
                ev = evd["ev"]
                if ev.get("type") == "q_done":
                    env = ev.get("envelope") or {}
                    qid = qids[i] if i < len(qids) else f"{iid}#{ev.get('index')}"
                    i += 1
                    verdict, issues = grade_env(qid, env, "gui")
                    rows.append({"qid": qid, "src": "gui", "verdict": verdict,
                                 "issues": issues, "status": env.get("status"),
                                 "trace": env.get("trace_id"),
                                 "ans_head": (env.get("answer") or "")[:70]})
        for evd in d.get("events", []):
            ev = evd["ev"]
            if ev.get("type") == "synthesis_done":
                syn = ev.get("synthesis") or {}
                if syn.get("status") not in ("ok", "partial"):
                    rows.append({"qid": f"{iid}-syn", "src": "gui", "verdict": "BUG",
                                 "issues": [f"synthesis status={syn.get('status')}"]})
    return rows


def grade_mcp():
    rows = []
    for p in sorted((ROOT / "mcp_out").glob("[SM]*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        iid = d["input_id"]
        env = d.get("envelope") or {}
        qmap = {"MB1": ["M01", "M02", "M03"], "MB2": ["M04", "M05", "M06"],
                "MB3": ["M07", "M08"], "MAB1": ["MA01", "MA02", "MA03"],
                "MAB2": ["MA04", "MA05", "MA06"], "MAB3": ["MA07", "MA08"]}
        qids = qmap.get(iid, [iid])
        per_q = (env.get("meta") or {}).get("per_question") or []
        if per_q and iid not in qmap and len(per_q) > 1:
            # 单输入拆多子答：金提示按合并答案判（ans/ans2 可分落不同子答）
            merged = {"status": "answered", "answer": "", "citations": [],
                      "formulas": [], "verification": {}}
            for sub in per_q:
                merged["answer"] += "\n" + (sub.get("answer") or "")
                merged["citations"] += sub.get("citations") or []
                merged["formulas"] += sub.get("formulas") or []
                merged["status"] = sub.get("status") or merged["status"]
            verdict, issues = grade_env(iid, merged, "mcp")
            rows.append({"qid": iid, "src": "mcp", "verdict": verdict,
                         "issues": issues, "status": merged["status"],
                         "ans_head": merged["answer"][:70]})
        elif per_q:
            for j, sub in enumerate(per_q):
                qid = qids[j] if j < len(qids) else f"{iid}#{j+1}"
                verdict, issues = grade_env(qid, sub, "mcp")
                rows.append({"qid": qid, "src": "mcp", "verdict": verdict,
                             "issues": issues, "status": sub.get("status"),
                             "ans_head": (sub.get("answer") or "")[:70]})
        else:
            verdict, issues = grade_env(qids[0], env, "mcp")
            rows.append({"qid": qids[0], "src": "mcp", "verdict": verdict,
                         "issues": issues, "status": env.get("status"),
                         "ans_head": (env.get("answer") or "")[:70]})
    return rows


if __name__ == "__main__":
    rows = grade_gui() + grade_mcp()
    (ROOT / "grades.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    tally = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("tally:", tally, "total:", len(rows))
    for r in rows:
        mark = {"OK": " ", "BUG": "✗", "REFUSED_OK": "✓"}[r["verdict"]]
        print(f"{mark} {r['src']:3} {r['qid']:5} {r['verdict']:10} {';'.join(r['issues'])[:100]}"
              f" | {r.get('ans_head','')[:40]}")
