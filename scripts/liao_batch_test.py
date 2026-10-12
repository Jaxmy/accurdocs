"""18 问一次（全新文档：辽宁电力市场规则）流式批量测试。

与上次批量（贷款/理财/绿色金融/公式/存款 5 份样例）完全不同——
全部问题取自辽宁规则 2085 块真实文档 + 1 题库外必拒答对照。
走 REST /v1/query/stream，按 q_done 逐题判分。
"""
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

QS = [
    # (qid, question, gold_regex on folded text, as_of)
    ("l01", "市场成员对结算查询结果有异议的，应在收到回复后多少个工作日内以书面方式提出？",
     r"10\s*个工作日", None),
    ("l02", "省内电力直接交易分为哪两种类型？", r"典型曲线交易.*分时段交易|分时段交易.*典型曲线交易", None),
    ("l03", "分时段交易将每天多少小时分为若干时段？", r"24\s*小时", None),
    ("l04", "月度、月内（含滚撮）交易的价格约束参照什么执行？", r"出清价格", None),
    ("l05", "批发用户的月度净合约量上限如何计算？", r"合同容量\s*[×xX*]\s*24", None),
    ("l06", "本省光伏项目在4、5、6、7、8月的可交易时间是几点到几点？",
     r"4[:：]00\s*[-–—~至]\s*19[:：]00", None),
    ("l07", "交易电量的最小申报单位是多少兆瓦时？", r"0\.001\s*兆瓦", None),
    ("l08", "非竞价电化学储能应在D-1日几点前申报D日充放电功率曲线与调频时段？",
     r"08[:：]15", None),
    ("l09", "系统负荷预测每天共计多少个点？", r"96\s*个点", None),
    ("l10", "批发用户任意时段申报量的理论最大用电能力按什么计算？", r"0\.25", None),
    ("l11", "火电机组每一个报价段的长度不能小于机组额定容量的多少？", r"5\s*%", None),
    ("l12", "火电机组申报价格的最小单位是多少？", r"10\s*元/兆瓦时", None),
    ("l13", "电化学储能的电能量充电、放电报价分别不高于几段？", r"7\s*段", None),
    ("l14", "燃煤机组冷态启动通知时间上限（E1）的暂定数值是多少？", r"48\s*小时", None),
    ("l15", "热态启动时大于60万千瓦机组的启动费用范围是多少？",
     r"5\s*[-–—~－至到]\s*50\s*万元", None),
    ("l16", "燃煤自备电厂的实际上网电量按照什么标准结算？", r"90\s*%", None),
    ("l17", "虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？",
     r"6\s*MW.*12\s*MWh|12\s*MWh.*6\s*MW", None),
    ("l18", "居民阶梯水价分为哪几档？", None, None),  # 库外问题 → 应拒答
]


def fold(s):
    return re.sub(r"\s+", "", s or "")


def grade(qid, answer, status):
    if qid == "l18":
        return "refused_ok" if status == "refused" else "miss_refusal"
    if status == "refused":
        return "refused"
    pat = next(p for i, _, p, _ in QS if i == qid)
    return "correct" if pat and re.search(pat, fold(answer)) else "wrong"


def main():
    body = {
        "query": "\n".join(f"{i}. {q}" for i, (_, q, _, _) in enumerate(QS, 1)),
        "as_of": "2025-12-01",
        "mode": "quote",
        "request_id": "liao_18q",
    }
    req = urllib.request.Request(
        BASE + "/v1/query/stream",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    qdones = {}
    n_delta = 0
    plan_total = None
    merged = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        buf = b""
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                for line in frame.decode("utf-8", "replace").splitlines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    typ = ev.get("type")
                    if typ == "plan":
                        plan_total = ev["total"]
                        print(f"plan: total={ev['total']}")
                    elif typ == "draft_delta":
                        n_delta += 1
                    elif typ == "q_done":
                        e = ev["envelope"]
                        qdones[ev["index"]] = (time.time() - t0, e)
                        ans = (e.get("answer") or (e.get("refusal") or {}).get("detail") or "")[:55]
                        print(f"q_done #{ev['index']:2d} @{time.time()-t0:5.1f}s "
                              f"{e['status']:8s} {ans}")
                    elif typ == "done":
                        merged = ev["envelope"]
                    elif typ == "error":
                        print("ERROR:", ev.get("message"))

    print(f"\ntotal={plan_total} draft_delta={n_delta} elapsed={time.time()-t0:.0f}s")
    out = []
    tally = {}
    for i, (qid, q, pat, _) in enumerate(QS, 1):
        ts, e = qdones.get(i, (0, {}))
        st = e.get("status", "?")
        ans = e.get("answer") or ""
        g = grade(qid, ans, st)
        tally[g] = tally.get(g, 0) + 1
        out.append({"qid": qid, "grade": g, "status": st, "at_s": round(ts, 1),
                    "ans": ans[:120], "q": q})
        print(f"{qid} [{st:8s}] {g:12s} @{ts:5.1f}s  {(ans or '')[:70]}")
    print("TALLY:", tally)
    with open("data/batch_test/liao_18q_results.json", "w", encoding="utf-8") as f:
        json.dump({"tally": tally, "items": out,
                   "elapsed_s": time.time() - t0,
                   "merged_status": (merged or {}).get("status"),
                   "merged_n_answered": (merged or {}).get("meta", {}).get("n_answered")},
                  f, ensure_ascii=False, indent=1)
    print("saved -> data/batch_test/liao_18q_results.json")


if __name__ == "__main__":
    main()
