"""测试战役 GUI 入口驱动：按 index.html askBody() 契约 plan→execute SSE，事件落盘。

用法：python scripts/campaign_gui.py [输入ID ...]   # 缺省跑全部 22 输入
"""
import json

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10"
INPUTS = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))


def post(path, body, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_execute(body, timeout=600):
    """POST /v1/query/execute SSE → [(t, ev)]（模板：scripts/stream_e2e.py）。"""
    req = urllib.request.Request(
        BASE + "/v1/query/execute", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                    events.append((round(time.time() - t0, 2), json.loads(line[6:])))
    return events


def run_one(item):
    iid, query, mode = item["id"], item["query"], item["mode"]
    out = ROOT / "gui_events" / f"{iid}.json"
    if out.exists():
        print(f"[{iid}] 已有记录，跳过")
        return
    body = {"query": query, "as_of": "", "mode": mode,
            "filters": {}, "require_review": False, "request_id": f"gui_{iid}"}
    record = {"input_id": iid, "query": query, "mode": mode, "t_start": time.time()}
    t0 = time.time()
    # 阶段1：/v1/plan（GUI 提问第一步）
    try:
        plan_resp = post("/v1/plan", body, timeout=300)
        record["plan"] = plan_resp
        qs = (plan_resp.get("plan") or {}).get("items") or []
        n = len(qs)
        # 拆分合理性 = 题干数下限（批量按编号题干计数——题干内"其中…？"不多计；
        # 单输入按问号数。多问句/混合题可再拆 reason 项，不做精确比对）
        import re as _re
        numbered = len(_re.findall(r"(?m)^\s*\d+[.、]", query))
        qmarks = numbered if numbered else query.count("？") + query.count("?")
        ok = n >= max(1, qmarks)
        flag = "OK" if ok else f"MISMATCH(items={n} < qmarks={qmarks})"
        print(f"[{iid}] plan: items={n} need_confirm={plan_resp.get('need_confirm')} {flag}")
        record["plan_split_ok"] = ok
    except Exception as e:  # noqa: BLE001
        record["plan_error"] = str(e)
        print(f"[{iid}] plan ERROR: {e}")
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    # 阶段2：/v1/query/execute（携带完整 plan，等价 GUI「全部执行」）
    try:
        events = stream_execute({**body, "plan": plan_resp.get("plan")})
        record["events"] = [{"t": t, "ev": ev} for t, ev in events]
        for t, ev in events:
            typ = ev.get("type")
            if typ == "q_done":
                e = ev.get("envelope") or {}
                print(f"[{iid}] q_done #{ev.get('index')} @{t}s status={e.get('status')} "
                      f"ans={(e.get('answer') or '')[:40].replace(chr(10), ' ')}")
            elif typ == "synthesis_done":
                syn = ev.get("synthesis") or {}
                print(f"[{iid}] synthesis_done @{t}s status={syn.get('status')}")
            elif typ == "error":
                print(f"[{iid}] ERROR ev: {ev.get('message')}")
        done = [ev for _, ev in events if ev.get("type") == "done"]
        if done:
            e = done[0].get("envelope") or {}
            print(f"[{iid}] done @{time.time()-t0:.0f}s status={e.get('status')} "
                  f"answered={e.get('meta', {}).get('n_answered')}/{e.get('meta', {}).get('n_questions')}")
    except Exception as e:  # noqa: BLE001
        record["exec_error"] = str(e)
        print(f"[{iid}] execute ERROR: {e}")
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{iid}] -> {out.name}")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for item in INPUTS:
        if only and item["id"] not in only:
            continue
        run_one(item)
