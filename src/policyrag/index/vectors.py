"""Embedding 抽象层：fastembed(ONNX) 优先，可换后端。

内存红线（计划 §二）：embedding 与 reranker 互斥加载（单飞锁在 pipeline 层）。
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

__all__ = ["Embedder", "FastEmbedEmbedder", "get_embedder"]

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_DIM = 512  # bge-small-zh-v1.5 实际输出 512 维（384 是 en 版的，别搞混）


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """fastembed ONNX 后端（int8 量化，峰值 <500MB）。"""

    def __init__(self, model_name: str = DEFAULT_MODEL, dim: int | None = None):
        from fastembed import TextEmbedding  # 懒加载，未装则报错清晰

        self._model = TextEmbedding(model_name, threads=2)
        self._lock = threading.Lock()
        # 实测探针维度——不同 bge 变体维度不同（zh=512/en=384），不硬编码
        probe = list(self._model.embed(["dim_probe"]))[0]
        self.dim = int(np.asarray(probe).shape[-1]) if dim is None else dim
        self.model_name = model_name
        # 向量库指纹用身份键：同后端换模型（同维度也）必须触发全量重嵌
        self.identity = f"fastembed:{model_name}:{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:  # 单飞：与 reranker 互斥由外层保证，内部也防并发
            vecs = list(self._model.embed(texts, batch_size=16))
        arr = np.asarray(vecs, dtype=np.float32)
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class HashEmbedder:
    """零依赖降级后端：字符 n-gram 哈希向量。

    只用于无 fastembed 环境的冒烟测试——检索质量远低于 bge，不进生产。
    """

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self.model_name = "hash-ngram"
        self.identity = f"hash:{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = text.lower()
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                h = hash(t[i : i + n]) % self.dim
                v[h] += 1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


_instance: Embedder | None = None
_inst_lock = threading.Lock()


def get_embedder(backend: str = "fastembed", model_name: str = DEFAULT_MODEL) -> Embedder:
    global _instance
    with _inst_lock:
        if _instance is None:
            if backend == "hash":
                _instance = HashEmbedder()
            else:
                try:
                    _instance = FastEmbedEmbedder(model_name)
                except Exception:
                    _instance = HashEmbedder()  # 降级：冒烟可用
        return _instance
