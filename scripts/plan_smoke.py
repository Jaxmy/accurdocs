"""M3/M4 live 冒烟：plan 生成 → 裁剪 → execute 两阶段（检索 + 综合推理）。"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTHONUTF8", "1")
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)

from policyrag.settings import make_llm, service_config
from policyrag.service.core import CoreService
from policyrag.models.retrieval import QueryRequest

svc = CoreService(make_llm(), service_config())

Q = ("1. 资本充足率不得低于多少？\n"
     "2. 逾期贷款罚息利率是多少？\n"
     "3. 结合前两问评估该办法对银行风险控制的影响。")

t0 = time.time()
req = QueryRequest(query=Q, as_of="2025-12-01", mode="quote", request_id="smoke_plan")
p = svc.build_plan(req)
print(f"== plan @{time.time()-t0:.1f}s need_confirm={p['need_confirm']} degraded={p['plan'].get('degraded')}")
for it in p["plan"]["items"]:
    print(f"   [{it['qid']}] {it['kind']:8s} spec={it['specificity']:8s} "
          f"vars={len(it['variants'])} deps={it['depends_on']} | {it['question'][:34]}")

print("== execute（流式）")
plan_dict = p["plan"]
from policyrag.query.search_plan import plan_from_dict
plan = plan_from_dict(plan_dict)
seen = []
for ev in svc.ask_stream(req, plan=plan):
    typ = ev.get("type")
    seen.append(typ)
    if typ == "q_done":
        e = ev["envelope"]
        print(f"   q_done #{ev.get('index')} {e['status']:8s} "
              f"{(e.get('answer') or '')[:48]}")
    elif typ == "synthesis_start":
        print(f"   synthesis_start basis={ev.get('n_basis')}")
    elif typ == "synthesis_done":
        s = ev["synthesis"]
        print(f"   synthesis_done status={s['status']}")
        print("   SYNTHESIS:", (s.get("text") or "")[:160])
    elif typ == "error":
        print("   ERROR:", ev.get("message"))
print(f"== total {time.time()-t0:.0f}s events={seen.count('draft_delta')} deltas")
print("   event order ok:",
      seen.index("synthesis_start") > seen.index("q_done") if "synthesis_start" in seen else "no-synthesis")
