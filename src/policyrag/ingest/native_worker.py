"""原生库子进程隔离（#15）：RapidOCR（onnxruntime）段错误只死子进程，不拖死 API。

背景：图表 OCR 曾把整个 uvicorn 进程段错误带走（exit 139）——原生崩溃穿透
Python try/except，唯一可靠的边界是进程边界。MinerU 已是外部 CLI（天然隔离）；
本模块隔离 RapidOCR：父进程 `ocr_isolated()` 起子进程跑 `python -m
policyrag.ingest.native_worker task.json result.json`，崩溃/超时返回空文本
并如实标注，绝不让服务陪葬。图片降采样护栏一并在子进程内做（大裁剪图是
原生分配器压力源）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["ocr_isolated", "main"]

_MAX_DIM = 1600   # 降采样护栏：长边超限先缩（数字表格 OCR 精度损失可忽略）


def _downscale(img_path: Path) -> str:
    """长边超限的图先缩（Pixmap.shrink 2^n 折半）——子进程内执行。"""
    try:
        import math
        import pymupdf
        pix = pymupdf.Pixmap(str(img_path))
        m = max(pix.width, pix.height)
        if m <= _MAX_DIM:
            return str(img_path)
        n = max(1, math.ceil(math.log2(m / _MAX_DIM)))
        pix.shrink(n)
        out = img_path.with_name(img_path.stem + "_sm.jpg")
        out.write_bytes(pix.tobytes("jpeg", jpg_quality=85))
        return str(out)
    except Exception:  # noqa: BLE001
        return str(img_path)


def main() -> None:
    """子进程入口：task.json → result.json。任何异常只写错误结果，不抛崩。"""
    task_path, result_path = Path(sys.argv[1]), Path(sys.argv[2])
    out: dict = {"ok": False, "data": None, "error": ""}
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("kind") == "ocr":
            from rapidocr import RapidOCR
            img = _downscale(Path(task["image"]))
            ocr = RapidOCR()
            res = ocr(img)
            out["data"] = [str(t) for t in
                           (list(res.txts) if res and res.txts is not None else [])]
            out["ok"] = True
        else:
            out["error"] = f"未知任务类型 {task.get('kind')!r}"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        result_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def ocr_isolated(image_path: str | Path, timeout: int = 120) -> list[str]:
    """子进程 RapidOCR：返回图内文字行；崩溃/超时/失败 → []（不抛、不拖死服务）。"""
    image_path = Path(image_path)
    if not image_path.exists():
        return []
    with tempfile.TemporaryDirectory(prefix="pr_ocr_") as td:
        tp, rp = Path(td) / "task.json", Path(td) / "result.json"
        tp.write_text(json.dumps({"kind": "ocr", "image": str(image_path)}),
                      encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "policyrag.ingest.native_worker",
                 str(tp), str(rp)],
                capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return []
        except OSError:
            return []
        if rp.exists():
            try:
                out = json.loads(rp.read_text(encoding="utf-8"))
                return list(out.get("data") or []) if out.get("ok") else []
            except Exception:  # noqa: BLE001
                return []
        # 子进程崩溃（段错误等）：result 未写出
        _ = proc.returncode
        return []


if __name__ == "__main__":
    main()
