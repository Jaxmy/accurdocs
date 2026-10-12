"""阅读器 REST 冒烟：上传超长条款 → fulltext 拼合块 → /v1/chunks/{id} block_index 对位。"""

import io
import uuid


class MockLLM:
    name = "mock"
    caps = type("C", (), {"structured_output": False, "json_mode": True})()

    def __init__(self, responses=None):
        self._r = list(responses or [])
        self.calls = 0

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None):
        from policyrag.llm.base import LLMResult
        from policyrag.models.answer import Usage
        self.calls += 1
        p = self._r.pop(0) if self._r else {}
        return LLMResult(text="{}", usage=Usage(), parsed=p, model="mock",
                         stop_reason="end_turn")


def _client():
    from fastapi.testclient import TestClient
    from policyrag.adapters.rest import app as appmod
    from policyrag.service.core import CoreService
    from policyrag.settings import service_config

    cfg = service_config()
    cfg.data_dir = __import__("pathlib").Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.embed_backend = "hash"
    appmod._svc = CoreService(MockLLM(), cfg)
    return TestClient(appmod.app)


def test_fulltext_merge_and_block_index():
    c = _client()
    # 超长第X条：>800 字触发 _split_long_text，产出 …/第二条 + …/第二条/续N
    long_text = "第二条 " + "省间日前差价电能量电费按合约电量与节点价差结算。" * 60
    content = f"# 测试规则\n\n第一条 总则。\n\n{long_text}\n\n第三条 附则。\n".encode()
    up = c.post("/v1/upload", files={"file": ("拼合测试.md", io.BytesIO(content),
                                              "text/markdown")}).json()
    assert up.get("doc_key"), up
    ft = c.get(f"/v1/documents/{up['doc_key']}/fulltext").json()
    assert "error" not in ft, ft
    merged = [b for b in ft["blocks"] if len(b.get("chunk_ids", [])) > 1]
    assert merged, f"超长条款应拼合出多成员块，实测 {[(b['chunk_id'], len(b.get('chunk_ids', []))) for b in ft['blocks']]}"
    b = merged[0]
    assert b["chunk_id"] == b["chunk_ids"][0]
    assert b["text"].count("\n") == len(b["pieces"]) - 1
    # 先拼合再分页：展示块数 < 原始 chunk 数
    rows = c.get("/v1/chunks_list", params={"doc_key": up["doc_key"], "limit": 500}).json()
    assert ft["total_blocks"] < len(rows), (ft["total_blocks"], len(rows))
    # 成员 id 的 block_index == 所属拼合块下标；旧字段齐全
    idx = ft["blocks"].index(b)
    for member in b["chunk_ids"]:
        ch = c.get(f"/v1/chunks/{member}").json()
        assert ch["block_index"] == idx, (member, ch["block_index"], idx)
        for key in ("chunk_id", "text", "clause_path", "doc_key", "page", "kind", "meta"):
            assert key in ch, key
    # 树绑定来源页区间（PDF 视图定位锚）：叶节点必有 page/page_end
    leaves = [n for n in ft["tree"] if n.get("chunk_id")]
    assert leaves and all(n.get("page") for n in leaves)
    assert all(n.get("page_end") for n in ft["tree"])
    c.delete("/v1/documents/" + up["doc_key"])
