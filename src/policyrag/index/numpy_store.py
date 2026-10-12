"""numpy memmap 向量存储——MVP 与 3.14 降级路径（计划 §二）。

数百份文档 <20 万 chunk × 384 维 float32 ≈ 300MB，暴力余弦完全够用。
全量 memmap 不吃堆内存。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["NumpyVectorStore"]


class NumpyVectorStore:
    """向量库：ids.json + vectors.f32 (memmap) + norms.f32。"""

    def __init__(self, root: str | Path, dim: int = 384):
        self.root = Path(root)
        self.dim = dim
        self.root.mkdir(parents=True, exist_ok=True)
        self._ids_path = self.root / "ids.json"
        self._vec_path = self.root / "vectors.f32"
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._mm: np.memmap | None = None

    # -- 写侧 ---------------------------------------------------------------

    def _release(self) -> None:
        """释放已打开的 memmap——Windows 上旧映射不关，重建文件会 OSError Errno 22。"""
        if self._mm is not None:
            try:
                self._mm._mmap.close()  # numpy memmap 底层 mmap 句柄
            except Exception:  # noqa: BLE001
                pass
            self._mm = None

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(f"vectors {vectors.shape} 与 ids {len(ids)} 不匹配")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"维度 {vectors.shape[1]} != 配置 {self.dim}")
        # 归一化（余弦 = 点积）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        self._release()  # 先关旧映射再重建（Windows 文件锁）
        self._ids_path.write_text(
            json.dumps(ids, ensure_ascii=False), encoding="utf-8"
        )
        if vectors.shape[0] == 0:
            self._vec_path.write_bytes(b"")
            self._ids = []
            self._index = {}
            return
        fp = np.memmap(self._vec_path, dtype=np.float32, mode="w+", shape=vectors.shape)
        fp[:] = vectors[:]
        fp.flush()
        del fp
        self._ids = list(ids)
        self._index = {cid: i for i, cid in enumerate(ids)}
        self._mm = None  # 重新打开

    def _open(self) -> np.memmap:
        if self._mm is None:
            if not self._ids_path.exists():
                self._ids = []
                self._index = {}
                raise FileNotFoundError(f"向量库未构建: {self.root}")
            ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            self._ids = ids
            self._index = {cid: i for i, cid in enumerate(ids)}
            if not ids:
                raise ValueError(f"向量库为空: {self.root}")
            # 一致性校验：ids 与向量文件行数不符（崩溃残留）→ 明确报错触发重建
            expect = len(ids) * self.dim * 4
            actual = self._vec_path.stat().st_size if self._vec_path.exists() else 0
            if actual != expect:
                raise ValueError(
                    f"向量库不一致(ids={len(ids)} expect={expect}B actual={actual}B)，需重建")
            self._mm = np.memmap(
                self._vec_path, dtype=np.float32, mode="r",
                shape=(len(self._ids), self.dim),
            )
        return self._mm

    # -- 读侧 ---------------------------------------------------------------

    @property
    def size(self) -> int:
        try:
            return len(self._open())
        except (FileNotFoundError, ValueError):
            return 0

    def search(self, query_vec: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        """暴力余弦 top-k。query_vec 会被归一化。空库返回 []。"""
        try:
            mm = self._open()
        except (FileNotFoundError, ValueError):
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        scores = mm @ q  # (N,)
        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        # 部分排序，避免全排开销
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def snapshot(self) -> tuple[list[str], np.ndarray]:
        """读出当前全部 id+向量（增量重建用）。空库返回 ([], zeros)。"""
        try:
            mm = self._open()
            return list(self._ids), np.asarray(mm[:])
        except (FileNotFoundError, ValueError):
            return [], np.zeros((0, self.dim), dtype=np.float32)

    def get(self, chunk_id: str) -> np.ndarray | None:
        mm = self._open()
        i = self._index.get(chunk_id)
        return np.asarray(mm[i]) if i is not None else None
