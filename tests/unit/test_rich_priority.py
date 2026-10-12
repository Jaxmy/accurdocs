"""富化插队队列回归（提问可插队）：优先级排序、批量让行、run_inline 完成信号。"""

from policyrag.ingest.enrichment import RichWorker, estimate_rich_seconds


class _FakeMeta:
    def __init__(self):
        self.jobs = {}
        self.updates = []

    def job_upsert(self, job_id, doc_key, kind, payload, status="queued",
                   progress=0.0, note=""):
        self.jobs[job_id] = {"job_id": job_id, "doc_key": doc_key, "kind": kind,
                             "payload": payload, "status": status,
                             "progress": progress, "note": note}

    def job_update(self, job_id, status, progress, note=""):
        self.updates.append((job_id, status, progress, note))
        if job_id in self.jobs:
            self.jobs[job_id].update(status=status, progress=progress, note=note)

    def job_get(self, job_id):
        return self.jobs.get(job_id)

    def job_list(self, doc_key=None, status=None):
        rows = list(self.jobs.values())
        if status:
            rows = [r for r in rows if r["status"] == status]
        return rows

    def get_chunk(self, cid):
        return None

    def commit(self):
        pass


def _worker(meta=None):
    return RichWorker(meta or _FakeMeta(), on_index_refresh=None)


def test_priority_order_and_p0_counter():
    w = _worker()
    w._prio.update({"b1": 1, "b2": 1, "ask": 0})
    w._enqueue("b1", 1)
    w._enqueue("b2", 1)
    w._enqueue("ask", 0)
    assert w._p0_waiting == 1
    assert w._should_yield("b1") is True     # 批量见到插队信号 → 让行
    assert w._should_yield("ask") is False   # 插队任务自身不让行
    assert w._pop() == "ask"
    assert w._p0_waiting == 0
    assert w._should_yield("b1") is False
    assert w._pop() == "b1"
    assert w._pop() == "b2"


def test_submit_priority_recorded():
    w = _worker()
    jid = w.submit("dk@v1", "dk", "v1", __file__, "chart",
                   chart_local_ids=["c1"], priority=0)
    job = w.meta.job_get(jid)
    assert job["payload"]["priority"] == 0
    assert w._prio[jid] == 0


def test_run_inline_waits_for_worker_and_signals():
    meta = _FakeMeta()
    w = _worker(meta)
    processed = []

    def fake_process(job_id):
        job = meta.job_get(job_id)
        if not job or job["status"] in ("done", "failed"):
            w._signal_done(job_id)     # 与真实 _process 同：重复入队项也发信号
            return
        processed.append(job_id)
        meta.job_update(job_id, "done", 1.0, "完成")
        w._signal_done(job_id)

    w._process = fake_process
    jid = w.submit("dk@v1", "dk", "v1", __file__, "chart",
                   chart_local_ids=["c1"], priority=1)
    w.run_inline(jid)                  # 提问插队：提到 0 优先级并等待完成
    assert processed == [jid]
    assert w._p0_waiting == 0


def test_yield_requeue_keeps_bulk_priority():
    w = _worker()
    w._prio["bulk"] = 1
    assert w._yield_requeue("bulk") is True
    assert w._prio["bulk"] == 1        # 让行后仍是批量优先级，不永久抬级
    w._prio["ask"] = 0
    w._enqueue("ask", 0)
    assert w._should_yield("bulk") is True


def test_estimate_rich_seconds():
    assert estimate_rich_seconds(0, 0) == 0
    assert estimate_rich_seconds(10, 0) == 80
    assert estimate_rich_seconds(0, 10, use_vision=True) == 110
    assert estimate_rich_seconds(1, 1) == 15   # 下限
