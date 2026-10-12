"""SSE 流式端点 E2E：多问题逐题完成逐题到达 + 单题 token 流。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def stream_ask(body, label):
    req = urllib.request.Request(
        BASE + "/v1/query/stream",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    events = []
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                    events.append((time.time() - t0, ev))
                    typ = ev.get("type")
                    if typ == "plan":
                        print(f"[{label}] plan: total={ev['total']}")
                    elif typ == "q_start":
                        print(f"[{label}] q_start #{ev['index']}: {ev['question'][:30]}")
                    elif typ == "draft_delta":
                        pass  # 高频，汇总后打
                    elif typ == "q_done":
                        e = ev["envelope"]
                        ans = (e.get("answer") or "")[:50].replace("\n", " ")
                        print(f"[{label}] q_done #{ev['index']} @{time.time()-t0:5.1f}s "
                              f"status={e['status']} ans={ans}")
                    elif typ == "done":
                        e = ev["envelope"]
                        print(f"[{label}] done @{time.time()-t0:5.1f}s status={e['status']} "
                              f"n_answered={e.get('meta', {}).get('n_answered')}")
                    elif typ == "error":
                        print(f"[{label}] ERROR: {ev.get('message')}")
                    elif typ == "phase":
                        pass
    n_delta = sum(1 for _, ev in events if ev.get("type") == "draft_delta")
    qdones = [ev for _, ev in events if ev.get("type") == "q_done"]
    # 逐题到达时间：验证"完成一题马上输出"
    for ev in qdones:
        ts = next(t for t, e in events if e is ev)
        print(f"[{label}]   q{ev['index']} 到达于 {ts:.1f}s")
    print(f"[{label}] draft_delta 事件数={n_delta} 总耗时={time.time()-t0:.1f}s")
    return events


if __name__ == "__main__":
    # 3 问小批量（跨文档）
    stream_ask({
        "query": "1. 资本充足率不得低于多少？\n2. 理财产品销售起点金额是多少？\n3. 房地产限购政策的具体条款是什么？",
        "as_of": "2025-12-01", "mode": "quote", "request_id": "e2e_stream_3q",
    }, "3Q")
    # 单问 token 流
    stream_ask({
        "query": "绿色项目单笔贷款额度上限是多少？",
        "as_of": "2025-12-01", "mode": "quote", "request_id": "e2e_stream_1q",
    }, "1Q")
