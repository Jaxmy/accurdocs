"""向量库指纹机制回归：embedding 身份变更必须全量重嵌（禁止混搭向量库）。"""

import os
import uuid

from policyrag.index.stores.sqlite_meta import MetaStore
from policyrag.service.core import CoreService, ServiceConfig


def _svc(tmp, backend="hash"):
    cfg = ServiceConfig(data_dir=f"{tmp}/data", trace_dir=f"{tmp}/traces",
                        embed_backend=backend)
    return CoreService(_llm(), cfg)


class _LLM:
    name = "mock"

    def complete(self, *a, **kw):
        raise AssertionError("测试不调用 LLM")


def _llm():
    return _LLM()


class TestVectorFingerprint:
    def test_identity_includes_model(self):
        """身份键含模型名/后端/维度——同后端换模型（同维度）也能识别。"""
        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        svc = _svc(tmp)
        key = svc._embed_key()
        assert svc.cfg.embed_backend in key
        assert str(svc._embedder.dim) in key

    def test_fingerprint_triggers_full_reembed(self):
        """指纹不匹配 → 全量重嵌（哪怕无新文档、仅查询路径触发）。"""
        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        svc = _svc(tmp)
        m = svc.meta
        m.upsert_document("甲", "v1", "甲办法")
        m.upsert_chunk("c1", "甲@v1", "p", "b", "资本充足率不得低于 8%", "资本充足率不得低于 8%", "text")
        m.commit()
        r1 = svc.rebuild_index()
        assert r1["full_reembed"] is True, "首次建档应全量嵌入"
        assert svc._fingerprint_key() == svc._embed_key()

        # 增量：同身份 → 复用旧向量，不全量
        m.upsert_chunk("c2", "甲@v1", "p2", "b2", "杠杆率不得超过 6%", "杠杆率不得超过 6%", "text")
        m.commit()
        r2 = svc.rebuild_index()
        assert r2["full_reembed"] is False and r2["embedded_new"] == 1

        # 模拟换模型/换后端：篡改指纹 → 首查路径（_ensure_index）强制全量重嵌
        svc._index = None
        svc._fingerprint_path().write_text('{"embed_key": "other-model:999"}',
                                           encoding="utf-8")
        svc._ensure_index()
        assert svc._fingerprint_key() == svc._embed_key(), \
            "查询路径必须自愈指纹（换 embedding 后不上传文档直接提问也不许混搭）"

    def test_new_docs_share_same_vectors(self):
        """新文档入库后与旧文档同一嵌入身份（同库同模型）。"""
        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        svc = _svc(tmp)
        m = svc.meta
        m.upsert_document("甲", "v1", "甲办法")
        m.upsert_chunk("c1", "甲@v1", "p", "b", "文本一", "文本一", "text")
        m.commit()
        svc.rebuild_index()
        key1 = svc._fingerprint_key()
        m.upsert_document("乙", "v1", "乙办法")
        m.upsert_chunk("c2", "乙@v1", "p", "b", "文本二", "文本二", "text")
        m.commit()
        svc.rebuild_index()
        assert svc._fingerprint_key() == key1
        assert not svc._fingerprint_key() != svc._embed_key()
