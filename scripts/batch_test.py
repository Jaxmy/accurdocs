"""批量提问 vs 逐题基线：检索覆盖率 + 推理正确率对照测试。"""
import sys, os, json, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTHONUTF8", "1")
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)

from policyrag.settings import make_llm, service_config
from policyrag.service.core import CoreService
from policyrag.models.retrieval import QueryRequest

QS = [
    ("q01", "资本充足率不得低于多少？", r"8\s*%", None),
    ("q02", "杠杆率不得超过多少？", r"6\s*%", None),
    ("q03", "信用贷款额度上限是多少？", r"1,?000\s*万", None),
    ("q04", "短期贷款的期限是多少？", r"一年", None),
    ("q05", "逾期贷款罚息利率是多少？", r"150\s*%", None),
    ("q06", "理财产品销售起点金额是多少？", r"1\s*元", None),
    ("q07", "私募理财产品合格投资者的金融资产标准是多少？", r"500\s*万", None),
    ("q08", "理财产品风险等级分为哪几级？", r"R1", None),
    ("q09", "绿色信贷余额占比不得低于多少？", r"15\s*%", None),
    ("q10", "绿色金融的不良率上限是多少？", r"3\s*%", None),
    ("q11", "绿色项目贷款期限最长多少年？", r"15\s*年?", None),
    ("q12", "绿色项目单笔贷款额度上限是多少？", r"5\s*亿", None),
    ("q13", "碳减排支持工具利率是多少？", r"1\.75\s*%", None),
    ("q14", "清洁能源项目的贴息比例和单笔上限是多少？", r"3\s*%.*2\s*亿|2\s*亿.*3\s*%", None),
    ("q15", "必开成本补偿费用公式的求和限元是怎么写的？", r"t\s*=\s*1", None),
    ("q16", "阻塞费用的计算公式是什么？", r"LMP", None),
    ("q17", "截至2024年6月1日，存款利率上限是多少？", r"2\s*%", "2024-06-01"),
    ("q18", "现行的存款利率上限是多少？", r"3\s*%", "2025-12-01"),
    ("q19", "房地产限购政策的具体条款是什么？", None, None),  # 应拒答
]
GOLD_KEYWORDS = {
    "q01": ["8%"], "q02": ["6%"], "q03": ["1,000万", "1000万"], "q04": ["一年"],
    "q05": ["150%"], "q06": ["1元"], "q07": ["500万"], "q08": ["R1"],
    "q09": ["15%"], "q10": ["3%"], "q11": ["15"], "q12": ["5亿"], "q13": ["1.75%"],
    "q14": ["3%", "2亿"], "q15": ["t=1"], "q16": ["LMP"], "q17": ["2%"], "q18": ["3%"],
}

def fold(s):
    return re.sub(r"\s+", "", s or "")

def grade(qid, answer, status):
    if qid == "q19":
        return "refused_ok" if status == "refused" else "miss_refusal"
    pats = next(p for i, _, p, _ in QS if i == qid)
    if status == "refused":
        return "refused"
    a = fold(answer)
    return "correct" if pats and re.search(pats, a) else "wrong"

def evidence_gold_hit(qid, trace_dir):
    kws = GOLD_KEYWORDS.get(qid, [])
    try:
        import glob
        f = sorted(glob.glob(trace_dir + "/step_*B3_B4*.json"))[0]
        ids = json.load(open(f, encoding="utf-8"))["output"]
    except Exception:
        return None
    hit = 0
    for cid in ids:
        row = svc.meta.get_chunk(cid)
        if row and any(fold(k) in fold(row["text"]) for k in kws):
            hit += 1
    return hit

svc = CoreService(make_llm(), service_config())
out = {"batch": [], "single": []}

# ---- 批量：19 问一次 ----
batch_q = "\n".join(f"{i+1}. {q}" for i, (_, q, _, _) in enumerate(QS))
t0 = time.time()
env = svc.ask(QueryRequest(query=chat_q if False else batch_q, as_of="2025-12-01",
                           mode="quote", request_id="batch_19q"))
print(f"[batch] {time.time()-t0:.0f}s status={env.status.value} ans_len={len(env.answer)}")
for qid, q, pat, _ in QS:
    g = grade(qid, env.answer, env.status.value)
    out["batch"].append({"qid": qid, "grade": g})
print("[batch] grades:", {x["qid"]: x["grade"] for x in out["batch"]})
json.dump({"batch_answer": env.answer, "batch_status": env.status.value},
          open("data/batch_test/batch_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 逐题基线 ----
for qid, q, pat, as_of in QS:
    t0 = time.time()
    e = svc.ask(QueryRequest(query=q, as_of=as_of or "2025-12-01", mode="quote",
                             request_id=f"single_{qid}"))
    g = grade(qid, e.answer, e.status.value)
    out["single"].append({"qid": qid, "grade": g, "sec": round(time.time()-t0, 1),
                          "answer": e.answer[:120]})
    print(f"[single] {qid} {g} ({time.time()-t0:.0f}s)")

json.dump(out, open("data/batch_test/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def tally(rows):
    from collections import Counter
    c = Counter(r["grade"] for r in rows)
    return dict(c)
print("BATCH :", tally(out["batch"]))
print("SINGLE:", tally(out["single"]))
