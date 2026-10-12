"""数值覆盖闸上线验收：l17 真实错题 + 回归抽查（防误杀）。"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTHONUTF8", "1")
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)

from policyrag.settings import make_llm, service_config
from policyrag.service.core import CoreService
from policyrag.models.retrieval import QueryRequest

CASES = [
    ("l17 (原错题·漏答)",
     "虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？"),
    ("l15 (回归·费用题)",
     "热态启动时大于60万千瓦机组的启动费用范围是多少？"),
    ("l04 (回归·非数值题)",
     "月度、月内（含滚撮）交易的价格约束参照什么执行？"),
    ("l03 (回归·时间题)",
     "分时段交易将每天多少小时分为若干时段？"),
]

svc = CoreService(make_llm(), service_config())
for label, q in CASES:
    t0 = time.time()
    env = svc.ask(QueryRequest(query=q, as_of="2025-12-01", mode="quote",
                               request_id="gate_" + str(abs(hash(q)) % 10**6)))
    ans = (env.answer or (env.refusal.detail if env.refusal else ""))[:90]
    print(f"[{label}] {time.time()-t0:4.1f}s status={env.status.value} "
          f"degr={env.degradations}")
    print(f"    {ans}")
