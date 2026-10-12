"""R2 战役出题落盘：r2_data → data/test_campaign_2026-10/r2/{questions.md,inputs.json}。"""
import json

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
from pathlib import Path

import r2_data as D

ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10" / "r2"
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "gui_events").mkdir(exist_ok=True)
(ROOT / "mcp_out").mkdir(exist_ok=True)

L = []
L.append("# policyrag R2 测试战役 · 提问表（2026-10-12）\n")
L.append("文档：《辽宁省电力市场运营基本规则及配套实施细则（试行4.0版征求意见稿）》"
         "（liaoning_market_rules@v1）+ 华创证券《寻找无边界扩张的行业》（@v1）。\n")
L.append("分组：①单一提问 ②组合提问 ③单一+分析 ④组合+分析，各 25 题共 100 题；"
         "考点与 R1 不重复、两文档内分散。金提示为 folded 正则（判分用），pages 为出处物理页。\n")
L.append("运行输入 70 个/端口：单题 50 + 批 18（B01-B09/G01-G09）+ 拒答探针 R01/R02（库外，不占 100 题表）。\n")
L.append("判分口径：拒答率/回答率（100 题表）、已答准确率=金提示全中题数/已答题数、"
         "召回率=金提示命中条数/金提示总条数、拒答正确率（探针）。"
         "GUI/MCP 双端口同题不一致记 bug。\n")

def row(qid, txt, golds, pages):
    g = "；".join(f"`{x}`" for x in golds)
    return f"| {qid} | {txt} | p{','.join(map(str, pages))} | {g} |\n"

L.append("\n## ① 单一提问 S01–S25（mode=quote）\n\n| qid | 提问 | 页 | 金提示 |\n|---|---|---|---|\n")
for q, t, g, p in D.GROUP1:
    L.append(row(q, t, g, p))

L.append("\n## ② 组合提问 M01–M25（mode=quote，B01–B09 批输入）\n")
for bid, items in D.GROUP2_BATCHES:
    L.append(f"\n**{bid}**：\n```\n")
    for i, (_, t, _, _) in enumerate(items, 1):
        L.append(f"{i}. {t}\n")
    L.append("```\n| qid | 提问 | 页 | 金提示 |\n|---|---|---|---|\n")
    for q, t, g, p in items:
        L.append(row(q, t, g, p))

L.append("\n## ③ 单一提问+分析 A01–A25（mode=analyze）\n\n| qid | 提问 | 页 | 金提示 |\n|---|---|---|---|\n")
for q, t, g, p in D.GROUP3:
    L.append(row(q, t, g, p))

L.append("\n## ④ 组合提问+分析 N01–N25（mode=analyze，G01–G09 批输入）\n")
for bid, items in D.GROUP4_BATCHES:
    L.append(f"\n**{bid}**：\n```\n")
    for i, (_, t, _, _) in enumerate(items, 1):
        L.append(f"{i}. {t}\n")
    L.append("```\n| qid | 提问 | 页 | 金提示 |\n|---|---|---|---|\n")
    for q, t, g, p in items:
        L.append(row(q, t, g, p))

L.append("\n## 拒答探针 R01–R02（库外，必拒）\n\n| qid | 提问 | mode | 期望 |\n|---|---|---|---|\n")
for q, t, mode in D.REFUSE_PROBES:
    L.append(f"| {q} | {t} | {mode} | **必拒**：refused、不编造 |\n")

(ROOT / "questions.md").write_text("".join(L), encoding="utf-8")
(ROOT / "inputs.json").write_text(
    json.dumps(D.build_inputs(), ensure_ascii=False, indent=1), encoding="utf-8")

qs = D.all_questions()
print(f"questions={len(qs)} inputs={len(D.build_inputs())} -> {ROOT}")
