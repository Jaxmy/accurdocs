"""R2-A 覆盖闸 A/B：question_kind 三模式（regex/llm/cross）离线回放对比。

用法：
  python scripts/r2_ab_gate.py label     # LLM 标注 100 题 + 20 探针（~15 次调用）
  python scripts/r2_ab_gate.py replay    # 冻结语料离线回放 → 两曲线 + 分歧矩阵

指标：
- 误拒率：完整类/非数值类上闸开火（armed 且 ok=False）的比例——必须 ≈0；
- 漏答拦截率：缺数值合成类上闸开火的比例——越高越好；
- 分歧矩阵：三模式 arming 在 120 题上的 ARMED/SOFT/OFF 分布（模式选择依据）。

回放零 LLM 成本：verify_numeric_coverage + arm_gate 是纯函数。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10" / "r2"
LABELS = ROOT / "qkind_labels.json"


def _fold(s):
    return re.sub(r"\s+", "", s or "")


# --------------------------------------------------------------- label
def do_label():
    import r2_data as D
    from policyrag.settings import make_llm
    from policyrag.query.search_plan import generate_plan

    questions = []
    qids = []
    for qid, q, *_ in D.all_questions():
        questions.append(q)
        qids.append(qid)
    for qid, q, _k, _d, _r in D.PROBE_QKIND:
        questions.append(q)
        qids.append(qid)

    llm = make_llm()
    labels = {}
    # generate_plan 自带 _BATCH=8 分片
    plan = generate_plan(llm, questions, mode="quote")
    for it in plan.items:
        labels[it.question] = {
            "question_kind": it.question_kind,
            "dimension": it.dimension,
            "intent": it.intent,
        }
    out = {}
    for qid, q in zip(qids, questions):
        out[qid] = labels.get(q, {"question_kind": None, "dimension": None, "intent": None})
    LABELS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_labeled = sum(1 for v in out.values() if v.get("question_kind"))
    print(f"labeled {n_labeled}/{len(out)} -> {LABELS}")


# -------------------------------------------------------------- replay
def _observed(qid):
    """R2 实测：(答案, 证据文本列表) — gui 优先，mcp 兜底；证据 id 取自 trace。"""
    import sqlite3
    conn = sqlite3.connect(str(_PROJ_ROOT / "data" / "meta.sqlite3"))
    conn.row_factory = sqlite3.Row
    ans, ev_ids = "", []
    for port, d in (("gui", ROOT / "gui_events"), ("mcp", ROOT / "mcp_out")):
        p = d / f"{qid}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        envs = []
        if port == "gui":
            envs = [e["ev"].get("envelope") or {}
                    for e in rec.get("events", []) if e["ev"].get("type") == "q_done"]
        else:
            envs = [rec.get("envelope") or {}]
        ans = "\n".join(e.get("answer") or "" for e in envs if e.get("status") in ("answered", "partial"))
        if ans:
            break
    for port in ("gui", "mcp"):
        for suffix in ("_1", ""):
            t = _PROJ_ROOT / "traces" / f"{port}_{qid}{suffix}" / "step_04_B3_B4_evidence.json"
            if t.exists():
                try:
                    ev_ids = json.loads(t.read_text(encoding="utf-8")).get("output") or []
                except Exception:  # noqa: BLE001
                    pass
                if ev_ids:
                    break
        if ev_ids:
            break
    ev_texts = []
    for cid in ev_ids:
        r = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
        if r:
            ev_texts.append(r["text"] or "")
    return ans, ev_texts


def do_replay():
    import r2_data as D
    from policyrag.verify.coverage import arm_gate, verify_numeric_coverage

    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    grades = json.loads((ROOT / "grades.json").read_text(encoding="utf-8")) if (ROOT / "grades.json").exists() else []
    gold_ok = {(g["qid"], g["src"]) for g in grades if g.get("verdict") == "OK"}

    rows = []
    # 100 题：分类 + 冻结语料
    for qid, q, golds, pages, grp in D.all_questions():
        ans, ev = _observed(qid)
        lab = labels.get(qid, {})
        regex_hit = bool(re.search(
            r"(多少|(?<![哪])几(?!乎|何)|数值|多大|多高|多长|多小|若干)", q))
        rows.append({"qid": qid, "q": q, "ans": ans, "ev": ev, "golds": golds,
                     "kind": lab.get("question_kind"), "dim": lab.get("dimension"),
                     "regex_hit": regex_hit, "probe": False})
    for qid, q, kind, dim, regex_hit in D.PROBE_QKIND:
        rows.append({"qid": qid, "q": q, "ans": "", "ev": [], "golds": [],
                     "kind": kind, "dim": dim, "regex_hit": regex_hit,
                     "probe": True, "expect_kind": kind, "expect_dim": dim})

    modes = ("regex", "llm", "cross")
    report = {"arming": {m: {"armed": 0, "soft": 0, "off": 0} for m in modes},
              "disagree": [], "false_refusal": {}, "interception": {},
              "llm_kind_acc": {"hit": 0, "n": 0}}
    for r in rows:
        states = {}
        for m in modes:
            st, sig = arm_gate(r["kind"], r["regex_hit"], m, r["dim"])
            states[m] = st
            report["arming"][m][st] += 1
        if len(set(states.values())) > 1:
            report["disagree"].append({"qid": r["qid"], "states": states,
                                       "kind": r["kind"], "regex_hit": r["regex_hit"]})
        if r["probe"]:
            report["llm_kind_acc"]["n"] += 1
            if r["kind"] == r.get("expect_kind"):
                report["llm_kind_acc"]["hit"] += 1

    # 闸结果回放（100 题有实测答案/证据的）
    for m in modes:
        fr_num = fr_non = n_num = n_non = 0
        inter = n_miss = 0
        for r in rows:
            if r["probe"] or not r["ans"] or not r["ev"]:
                continue
            kind, dim = r["kind"], r["dim"]
            st, _ = arm_gate(kind, r["regex_hit"], m, dim)
            # 完整类（实测答案，闸须放行）
            cov = verify_numeric_coverage(r["q"], r["ans"], r["ev"],
                                          question_kind=kind, dimension=dim, mode=m)
            if r["regex_hit"] or st == "armed":
                n_num += 1
                if not cov.ok:
                    fr_num += 1
            else:
                n_non += 1
                if not cov.ok:
                    fr_non += 1
            # 缺数值合成类：剥离金提示数值 span 后闸须开火
            if r["golds"] and (r["regex_hit"] or st == "armed"):
                stripped = r["ans"]
                for g in r["golds"]:
                    try:
                        stripped = re.sub(g, "", stripped)
                    except re.error:
                        pass
                cov2 = verify_numeric_coverage(r["q"], stripped, r["ev"],
                                               question_kind=kind, dimension=dim, mode=m)
                n_miss += 1
                if not cov2.ok:
                    inter += 1
        report["false_refusal"][m] = {
            "numeric_class": f"{fr_num}/{n_num}",
            "non_numeric_class": f"{fr_non}/{n_non}",
        }
        report["interception"][m] = f"{inter}/{n_miss}"

    out = ROOT / "ab_gate_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if cmd == "label":
        do_label()
    else:
        do_replay()
