"""R2 判分与指标：gui_events/ + mcp_out/ → grades.json + report.md。

指标（用户口径）：
- 拒答率 = 100 题表内被拒答题数 / 100（含误拒；探针另计）
- 回答率 = 已答题数 / 100
- 已答准确率 = 金提示全中题数 / 已答题数
- 召回率 = 金提示命中条数 / 金提示总条数（未答/拒答记 0 命中）
- 拒答正确率 = 探针正确拒答 / 探针数
判定 OK / BUG / REFUSED_OK；GUI/MCP 同题结论不一致记 bug。
"""
import json

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import r2_data as D  # noqa: E402

ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10" / "r2"
ID_RE = re.compile(r"\[[0-9a-fA-F]{16}\]")
LATEX_RE = re.compile(r"\\frac|\\sum|\\left|\\right|_\{|\^\{|【LaTeX】|\\mathrm")
GOLD = {q: (txt, golds, pages, grp) for q, txt, golds, pages, grp in D.all_questions()}
REFUSE = {q for q, _, _ in D.REFUSE_PROBES}


def fold(s):
    return re.sub(r"\s+", "", s or "")


def grade_env(qid, env):
    issues = []
    st = env.get("status")
    if qid in REFUSE:
        if st == "refused":
            return "REFUSED_OK", issues, 0, 0
        issues.append(f"应拒答实际 {st}")
        return "BUG", issues, 0, 0
    if qid not in GOLD:
        issues.append("题号不在题库")
        return "BUG", issues, 0, 0
    txt, golds, pages, grp = GOLD[qid]
    if st not in ("answered", "partial"):
        det = (env.get("refusal") or {}).get("detail", "")
        issues.append(f"未回答 status={st} {det[:50]}")
        return "BUG", issues, 0, len(golds)
    ans = env.get("answer") or ""
    f = fold(ans)
    fq = fold(txt)                      # 题面本身（前提型金提示不作答要求）
    hit = 0
    for g in golds:
        try:
            if re.search(fold(g), fq):
                hit += 1               # 金提示在题面里=用户前提，回述非必须
                continue
        except re.error:
            pass
        if re.search(fold(g), f):
            hit += 1
        else:
            issues.append(f"金提示未中: {g}")
    # 引用页码
    cpages = set()
    for c in env.get("citations") or []:
        d = c.get("display") or c
        if d.get("page"):
            cpages.add(d["page"])
    if pages and not (cpages & set(pages)):
        issues.append(f"引用页 {sorted(cpages)[:8]} 不含期望 p{pages}")
    # 质量红线
    if ID_RE.search(ans):
        issues.append("答案残留原子 ID")
    if LATEX_RE.search(ans):
        issues.append("答案含 LaTeX 片段")
    if "转写进行中" in ans:
        issues.append("答案含'转写进行中'占位")
    for fl in (env.get("verification") or {}).get("failures") or []:
        if "已删除" in fl:
            issues.append(f"校验删句: {fl[:40]}")
    if grp in (3, 4) and not (env.get("synthesis") or env.get("meta", {}).get("per_question")):
        pass  # analyze 的 synthesis 在 GUI 事件层核对
    verdict = "OK" if not issues else "BUG"
    return verdict, issues, hit, len(golds)


def merge_envs(envs):
    """合并池（R2-11 F5d）：池内全拒才标 refused，否则按合并答案判金提示——
    部分子答拒答不得污染其余子题判定（假拒答根因）。"""
    merged = {"status": "answered", "answer": "", "citations": [],
              "formulas": [], "verification": {}, "refusal": None}
    statuses = []
    for env in envs:
        merged["answer"] += "\n" + (env.get("answer") or "")
        merged["citations"] += env.get("citations") or []
        merged["formulas"] += env.get("formulas") or []
        statuses.append(env.get("status") or "answered")
        if env.get("refusal") and merged["refusal"] is None:
            merged["refusal"] = env.get("refusal")
    if statuses and all(s == "refused" for s in statuses):
        merged["status"] = "refused"
    elif statuses and all(s in ("refused", "partial") for s in statuses):
        merged["status"] = "partial"
    else:
        merged["status"] = "answered"
    return merged


def collect(port):
    """→ {qid: (env, trace)}（单输入拆多子答时合并；批输入按序映射）。"""
    qmap = D.input_qids()
    out = {}
    dirn = ROOT / ("gui_events" if port == "gui" else "mcp_out")
    for p in sorted(dirn.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        iid = d["input_id"]
        if d.get("plan_error") or d.get("exec_error") or d.get("error"):
            for qid in qmap.get(iid, [iid]):
                out[qid] = ({"status": "error",
                             "answer": "",
                             "refusal": {"detail": d.get("plan_error") or d.get("exec_error") or d.get("error")}},
                            None)
            continue
        if port == "gui":
            qdones = [evd["ev"].get("envelope") or {}
                      for evd in d.get("events", [])
                      if evd["ev"].get("type") == "q_done"]
            syn = [evd["ev"].get("synthesis") or {}
                   for evd in d.get("events", [])
                   if evd["ev"].get("type") == "synthesis_done"]
        else:
            env = d.get("envelope") or {}
            pq = (env.get("meta") or {}).get("per_question") or []
            qdones = pq if pq else [env]
            syn = [env.get("synthesis") or {}] if env.get("synthesis") else []
        qids = qmap.get(iid, [iid])
        if len(qdones) != len(qids):
            # 拆分数与题数不符（单输入拆多子答 / 批内过度拆分）：合并全池按题判分
            # （ans/金提示可分落不同子答——R1 实测坑）
            merged = merge_envs(qdones)
            if syn:
                merged["synthesis"] = syn[0]
            for qid in qids:
                out[qid] = (merged, (qdones[-1] or {}).get("trace_id") if qdones else None)
        else:
            for i, env in enumerate(qdones):
                qid = qids[i]
                if syn and i == len(qdones) - 1:
                    env = dict(env)
                    env["synthesis"] = syn[0]
                out[qid] = (env, env.get("trace_id"))
    return out


def grade_port(port):
    envs = collect(port)
    rows = []
    for qid, txt, golds, pages, grp in D.all_questions():
        env, trace = envs.get(qid, ({"status": "missing", "answer": "",
                                     "refusal": {"detail": "无记录"}}, None))
        verdict, issues, hit, total = grade_env(qid, env)
        rows.append({"qid": qid, "src": port, "group": grp, "verdict": verdict,
                     "status": env.get("status"), "issues": issues,
                     "gold_hit": hit, "gold_total": total, "trace": trace,
                     "ans_head": (env.get("answer") or "")[:60]})
    for qid in sorted(REFUSE):
        env, trace = envs.get(qid, ({"status": "missing", "answer": "",
                                     "refusal": {"detail": "无记录"}}, None))
        verdict, issues, hit, total = grade_env(qid, env)
        rows.append({"qid": qid, "src": port, "group": 0, "verdict": verdict,
                     "status": env.get("status"), "issues": issues,
                     "gold_hit": 0, "gold_total": 0, "trace": trace,
                     "ans_head": (env.get("answer") or "")[:60]})
    return rows


def metrics(rows):
    main = [r for r in rows if r["group"] in (1, 2, 3, 4)]
    probes = [r for r in rows if r["group"] == 0]
    answered = [r for r in main if r["status"] == "answered"]
    refused = [r for r in main if r["status"] == "refused"]
    full_ok = [r for r in answered if r["gold_hit"] == r["gold_total"] and r["gold_total"]]
    hit = sum(r["gold_hit"] for r in main)
    total = sum(r["gold_total"] for r in main)
    probe_ok = [r for r in probes if r["verdict"] == "REFUSED_OK"]
    n = len(main) or 1
    return {
        "n_main": len(main),
        "answered": len(answered),
        "refused": len(refused),
        "other": len(main) - len(answered) - len(refused),
        "answer_rate": round(len(answered) / n, 4),
        "refuse_rate": round(len(refused) / n, 4),
        "precision_answered": round(len(full_ok) / len(answered), 4) if answered else 0.0,
        "recall_gold": round(hit / total, 4) if total else 0.0,
        "recall_question": round(len(full_ok) / n, 4),
        "gold_hit": hit, "gold_total": total,
        "n_probe": len(probes),
        "refuse_probe_ok": len(probe_ok),
        "refuse_probe_rate": round(len(probe_ok) / len(probes), 4) if probes else 0.0,
    }


if __name__ == "__main__":
    g_rows = grade_port("gui")
    m_rows = grade_port("mcp")
    rows = g_rows + m_rows
    (ROOT / "grades.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = ["# R2 判分报告\n"]
    for port, pr in (("gui", g_rows), ("mcp", m_rows)):
        m = metrics(pr)
        rep.append(f"\n## {port.upper()}\n")
        rep.append(f"- 回答率 {m['answered']}/{m['n_main']} = {m['answer_rate']:.1%}；"
                   f"拒答率 {m['refused']}/{m['n_main']} = {m['refuse_rate']:.1%}"
                   f"（其他/异常 {m['other']}）\n")
        rep.append(f"- 已答准确率（金全中） {m['precision_answered']:.1%}"
                   f"（{len([r for r in pr if r['group'] in (1,2,3,4) and r['status']=='answered' and r['gold_hit']==r['gold_total']])}"
                   f"/{m['answered']}）\n")
        rep.append(f"- 召回率（金提示条） {m['gold_hit']}/{m['gold_total']} = {m['recall_gold']:.1%}；"
                   f"题级召回 {m['recall_question']:.1%}\n")
        rep.append(f"- 拒答探针 {m['refuse_probe_ok']}/{m['n_probe']} = {m['refuse_probe_rate']:.1%}\n")
        tally = {}
        for r in pr:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        rep.append(f"- 判定分布 {tally}\n")
    # 双端口一致性
    gm = {r["qid"]: r for r in g_rows}
    mm = {r["qid"]: r for r in m_rows}
    diff = [q for q in gm if q in mm and gm[q]["verdict"] != mm[q]["verdict"]]
    rep.append(f"\n## 双端口不一致（记 bug）：{len(diff)}\n")
    for q in diff:
        rep.append(f"- {q}: gui={gm[q]['verdict']}({gm[q]['status']}) "
                   f"mcp={mm[q]['verdict']}({mm[q]['status']})\n")
    (ROOT / "report.md").write_text("".join(rep), encoding="utf-8")

    print("".join(rep))
    print("\n== BUG 清单 ==")
    for r in rows:
        if r["verdict"] == "BUG":
            print(f"{r['src']:3} {r['qid']:5} {r['status']} :: "
                  f"{';'.join(r['issues'])[:110]}")
