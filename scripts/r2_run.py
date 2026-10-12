"""R2 战役双入口驱动：GUI（plan→execute SSE）与 MCP（stdio policy_ask）。

用法：
  python scripts/r2_run.py gui [ID ...]     # GUI 入口，缺省全部 70 输入
  python scripts/r2_run.py mcp [ID ...]     # MCP 入口
  python scripts/r2_run.py all [ID ...]     # 先 GUI 后 MCP
已有记录自动跳过（断点续跑）。输出 r2/gui_events/{ID}.json、r2/mcp_out/{ID}.json。
"""
import asyncio

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import r2_data as D  # noqa: E402

BASE = "http://127.0.0.1:8000"
ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10" / "r2"
INPUTS = D.build_inputs()


def post(path, body, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_execute(body, timeout=900):
    """POST /v1/query/execute SSE → [(t, ev)]。"""
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


def gui_one(item):
    iid, query, mode = item["id"], item["query"], item["mode"]
    out = ROOT / "gui_events" / f"{iid}.json"
    if out.exists():
        print(f"[gui {iid}] 已有记录，跳过", flush=True)
        return
    body = {"query": query, "as_of": "", "mode": mode,
            "filters": {}, "require_review": False, "request_id": f"gui_{iid}"}
    record = {"input_id": iid, "query": query, "mode": mode, "t_start": time.time()}
    t0 = time.time()
    try:
        plan_resp = post("/v1/plan", body, timeout=300)
        record["plan"] = plan_resp
        qs = (plan_resp.get("plan") or {}).get("items") or []
        numbered = len(re.findall(r"(?m)^\s*\d+[.、]", query))
        qmarks = numbered if numbered else query.count("？") + query.count("?")
        ok = len(qs) >= max(1, qmarks)
        record["plan_split_ok"] = ok
        print(f"[gui {iid}] plan items={len(qs)} need={max(1, qmarks)} "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
    except Exception as e:  # noqa: BLE001
        record["plan_error"] = str(e)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[gui {iid}] plan ERROR: {e}", flush=True)
        return
    try:
        events = stream_execute({**body, "plan": plan_resp.get("plan")})
        record["events"] = [{"t": t, "ev": ev} for t, ev in events]
        for t, ev in events:
            typ = ev.get("type")
            if typ == "q_done":
                e = ev.get("envelope") or {}
                print(f"[gui {iid}] q_done #{ev.get('index')} @{t}s "
                      f"status={e.get('status')} "
                      f"ans={(e.get('answer') or '')[:36].replace(chr(10), ' ')}",
                      flush=True)
            elif typ == "synthesis_done":
                print(f"[gui {iid}] synthesis @{t}s "
                      f"status={(ev.get('synthesis') or {}).get('status')}", flush=True)
            elif typ == "error":
                print(f"[gui {iid}] ERROR ev: {ev.get('message')}", flush=True)
    except Exception as e:  # noqa: BLE001
        record["exec_error"] = str(e)
        print(f"[gui {iid}] execute ERROR: {e}", flush=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gui {iid}] done {time.time() - t0:.0f}s -> {out.name}", flush=True)


async def mcp_all(only: set):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "policyrag.adapters.mcp.server"],
        env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
             "PATH": os.environ.get("PATH", "")})

    def text_of(result) -> str:
        return "\n".join(c.text for c in (getattr(result, "content", []) or [])
                         if getattr(c, "type", "") == "text")

    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            print("mcp tools:", [t.name for t in tools.tools], flush=True)
            for item in INPUTS:
                if only and item["id"] not in only:
                    continue
                iid = item["id"]
                out = ROOT / "mcp_out" / f"{iid}.json"
                if out.exists():
                    print(f"[mcp {iid}] 已有记录，跳过", flush=True)
                    continue
                t0 = time.time()
                try:
                    res = await sess.call_tool("policy_ask", {
                        "query": item["query"], "mode": item["mode"]})
                    txt = text_of(res)
                    try:
                        env = json.loads(txt)
                    except Exception:  # noqa: BLE001
                        env = {"raw": txt}
                    rec = {"input_id": iid, "query": item["query"],
                           "mode": item["mode"], "envelope": env, "text_len": len(txt)}
                    st = env.get("status", "?")
                    ans = (env.get("answer") or "")[:36].replace("\n", " ")
                    n_id = len(re.findall(r"\[[0-9a-fA-F]{16}\]", env.get("answer") or ""))
                    print(f"[mcp {iid}] {time.time() - t0:.0f}s status={st} "
                          f"ans={ans} id_marks={n_id}", flush=True)
                except Exception as e:  # noqa: BLE001
                    rec = {"input_id": iid, "query": item["query"],
                           "mode": item["mode"], "error": str(e)}
                    print(f"[mcp {iid}] ERROR: {e}", flush=True)
                out.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    ports = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = set(sys.argv[2:])
    sel = [i for i in INPUTS if not only or i["id"] in only]
    if ports in ("gui", "all"):
        for item in sel:
            gui_one(item)
    if ports in ("mcp", "all"):
        asyncio.run(mcp_all({i["id"] for i in sel}))
