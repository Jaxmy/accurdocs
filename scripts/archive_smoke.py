"""归档链路 + 用户原题回归（ID 隐藏 / 状态标签）冒烟。"""
import json
import re
import time
import urllib.parse
import urllib.request

B = "http://127.0.0.1:8000"


def get(p):
    return json.load(urllib.request.urlopen(B + p, timeout=120))


def post(p, body):
    req = urllib.request.Request(
        B + p, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=900))


print("== 1) 一键命名归档（勾选辽宁文档 → 电力市场）")
docs = get("/v1/documents")
liao = next(d for d in docs if "辽宁" in d["title"])
r = post("/v1/libraries/assign", {
    "library_name": "电力市场", "doc_keys": [liao["doc_key"]]})
print("   ", r)

print("== 2) AI 自动归档（未归档文档）")
t0 = time.time()
r = post("/v1/libraries/auto_archive", {"scope": "unarchived"})
print(f"    assigned={r.get('assigned')} created={r.get('libraries_created')} "
      f"candidates={r.get('candidates')} @{time.time()-t0:.0f}s")
print("    libraries:", [(l["name"], l["n_docs"]) for l in get("/v1/libraries")])

print("== 3) 用户原题回归：答案不得含原子编码")
Q = "电能量计量装置应该配置的电表等检测装置应该遵守什么标准？"
p = post("/v1/plan", {"query": Q, "as_of": "2025-12-01"})
plan = p["plan"]
req = urllib.request.Request(
    B + "/v1/query/execute",
    data=json.dumps({"query": Q, "as_of": "2025-12-01", "plan": plan}).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST")
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as resp:
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
                t = ev.get("type")
                if t == "q_done":
                    e = ev["envelope"]
                    ans = e.get("answer") or ""
                    print(f"    q_done @{time.time()-t0:.0f}s status={e['status']}")
                    print("    answer 有原子编码:", bool(re.search(r"\[[0-9a-f]{16}\]", ans)))
                    print("    line_refs 条数:", len(e.get("line_refs") or []))
                    print("    citations display/ref 分层:",
                          bool(e.get("citations")) and "display" in e["citations"][0])
                    print("    答案首句:", ans[:60])
                elif t == "error":
                    print("    ERROR:", ev.get("message"))
