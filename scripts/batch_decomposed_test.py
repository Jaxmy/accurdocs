"""批量提问（拆分后逐题）基准：19 问一次送入，按金标准逐题判分。

对照：旧单发批量 2/19；逐题基线 17/19。
"""
import json
import re
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTHONUTF8", "1")
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)

from policyrag.settings import make_llm, service_config
from policyrag.service.core import CoreService
from policyrag.models.retrieval import QueryRequest

QS = [
    ("q01", "资本充足率不得低于多少？", r"8\s*%"),
    ("q02", "杠杆率不得超过多少？", r"6\s*%"),
    ("q03", "信用贷款额度上限是多少？", r"1,?000\s*万"),
    ("q04", "短期贷款的期限是多少？", r"一年"),
    ("q05", "逾期贷款罚息利率是多少？", r"150\s*%"),
    ("q06", "理财产品销售起点金额是多少？", r"1\s*元"),
    ("q07", "私募理财产品合格投资者的金融资产标准是多少？", r"500\s*万"),
    ("q08", "理财产品风险等级分为哪几级？", r"R1"),
    ("q09", "绿色信贷余额占比不得低于多少？", r"15\s*%"),
    ("q10", "绿色金融的不良率上限是多少？", r"3\s*%"),
    ("q11", "绿色项目贷款期限最长多少年？", r"15\s*年?"),
    ("q12", "绿色项目单笔贷款额度上限是多少？", r"5\s*亿"),
    ("q13", "碳减排支持工具利率是多少？", r"1\.75\s*%"),
    ("q14", "清洁能源项目的贴息比例和单笔上限是多少？", r"3\s*%.*2\s*亿|2\s*亿.*3\s*%"),
    ("q15", "必开成本补偿费用公式的求和限元是怎么写的？", r"t\s*=\s*1"),
    ("q16", "阻塞费用的计算公式是什么？", r"LMP"),
    ("q17", "截至2024年6月1日，存款利率上限是多少？", r"2\s*%"),
    ("q18", "现行的存款利率上限是多少？", r"3\s*%"),
    ("q19", "房地产限购政策的具体条款是什么？", None),
]


def fold(s):
    return re.sub(r"\s+", "", s or "")


def grade(qid, answer, status):
    if qid == "q19":
        return "refused_ok" if status == "refused" else "miss_refusal"
    if status == "refused":
        return "refused"
    pat = next(p for i, _, p in QS if i == qid)
    return "correct" if pat and re.search(pat, fold(answer)) else "wrong"


svc = CoreService(make_llm(), service_config())
batch_q = "\n".join(f"{i+1}. {q}" for i, (_, q, _) in enumerate(QS))
t0 = time.time()

per = []
env = svc.ask(QueryRequest(query=batch_q, as_of="2025-12-01", mode="quote",
                           request_id="batch_decomposed"))
dt = time.time() - t0

pq = (env.meta or {}).get("per_question") or []
print(f"[batch-decomposed] {dt:.0f}s status={env.status.value} "
      f"n={len(pq)} n_answered={env.meta.get('n_answered')}")
out = []
for i, (qid, q, _) in enumerate(QS):
    e = pq[i] if i < len(pq) else {}
    ans = e.get("answer") or ""
    st = e.get("status") or "?"
    g = grade(qid, ans, st)
    out.append({"qid": qid, "grade": g, "status": st, "ans": ans[:80]})
    print(f"  {qid} [{st:8s}] {g:12s} {(ans or (e.get('refusal') or {}).get('detail', ''))[:60]}")

tally = {}
for x in out:
    tally[x["grade"]] = tally.get(x["grade"], 0) + 1
print("TALLY:", tally)
with open("data/batch_test/decomposed_results.json", "w", encoding="utf-8") as f:
    json.dump({"tally": tally, "items": out, "elapsed_s": dt,
               "merged_status": env.status.value}, f, ensure_ascii=False, indent=1)
print("saved -> data/batch_test/decomposed_results.json")
