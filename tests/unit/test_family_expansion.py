"""证据家族扩展回归（#9）：标题块命中补子条款块，防空心证据误拒。"""

from policyrag.models.retrieval import Evidence
from policyrag.query.pipeline import PipelineDeps, QueryPipeline


class _FakeLLM:
    name = "fake"
    caps = type("C", (), {"structured_output": False, "json_mode": True})()

    def complete(self, *a, **k):
        raise AssertionError("不应调用 LLM")


def _pipe(live_family):
    deps = PipelineDeps(index=None, llm=_FakeLLM(), registry={},
                        meta_lookup={}, corpus_version="t",
                        live_family=live_family)
    return QueryPipeline(deps, trace_root="tests/.tmp/family_traces")


def test_title_block_expands_children():
    kids = [
        {"chunk_id": "c1", "text": "依法取得发电项目核准或者备案文件，依法取得电力业务许可证。" * 2,
         "breadcrumb": "第三章 第七条 （一）", "page": 270},
        {"chunk_id": "c2", "text": "已与电网企业签订并并网调度协议，接入电力调度自动化系统。" * 2,
         "breadcrumb": "第三章 第七条 （二）", "page": 270},
    ]
    p = _pipe(lambda cid: kids if cid == "t1" else [])
    evs = [Evidence(chunk_id="t1", text="发电企业基本条件", doc_title="注册细则",
                    doc_version="v1", clause_path="第三章 第七条", page=270)]
    out = p._expand_families(evs)
    assert [e.chunk_id for e in out] == ["t1", "c1", "c2"]
    assert out[1].page == 270


def test_substantive_block_not_expanded():
    p = _pipe(lambda cid: [{"chunk_id": "x", "text": "y" * 50, "breadcrumb": "", "page": 1}])
    evs = [Evidence(chunk_id="t1", text="经营主体应当是财务独立核算、能够独立承担民事责任的企业。",
                    doc_title="T", doc_version="v1", clause_path="第六条", page=1)]
    out = p._expand_families(evs)
    assert [e.chunk_id for e in out] == ["t1"], "有实质句读的块不该扩展"


def test_dedupe_and_cap():
    kids = [{"chunk_id": f"k{i}", "text": "x" * 41, "breadcrumb": "", "page": 1}
            for i in range(10)]
    p = _pipe(lambda cid: kids)
    evs = [Evidence(chunk_id="t1", text="标题", doc_title="T", doc_version="v1",
                    clause_path="X", page=1)]
    out = p._expand_families(evs)
    assert len(out) <= 13, "扩展上限（12 子块）"
    assert "k0" in [e.chunk_id for e in out]

    # 已在证据中的子块不重复
    evs2 = [Evidence(chunk_id="t1", text="标题", doc_title="T", doc_version="v1",
                     clause_path="X", page=1),
            Evidence(chunk_id="k0", text="x" * 41, doc_title="T", doc_version="v1",
                     clause_path="X", page=1)]
    out2 = p._expand_families(evs2)
    ids = [e.chunk_id for e in out2]
    assert ids.count("k0") == 1


def test_core_live_family():
    from policyrag.index.stores.sqlite_meta import MetaStore
    from policyrag.service.core import CoreService

    m = MetaStore("tests/.tmp/family_meta.sqlite3")
    m.upsert_document("注册细则", "v1", "注册细则")
    m.upsert_chunk("t1", "注册细则@v1", "第三章/第七条", "第三章 第七条",
                   "发电企业基本条件", "发电企业基本条件", "article", page=270)
    m.upsert_chunk("i1", "注册细则@v1", "第三章/第七条/（一）", "第三章 第七条 （一）",
                   "依法取得发电项目核准或者备案文件。" * 3,
                   "依法取得发电项目核准或者备案文件。", "item", page=270)
    m.commit()
    svc = CoreService.__new__(CoreService)   # 只用 meta，不建索引
    svc.meta = m
    kids = svc.live_family("t1")
    assert [k["chunk_id"] for k in kids] == ["i1"]
    assert svc.live_family("missing") == []
