"""MinerU 引擎自管理：按需启动 / 配置自愈 / 闲置释放——复杂度对调用方完全隐藏。

调用方（formulas.run_mineru）只需 ensure_mineru_ready()；
引擎未起则自动拉起（mineru server + mineru-api），配置缺失自动补，
闲置 N 分钟自动释放（仅管理自己拉起的实例，不动外部启动的）。
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

__all__ = ["ensure_mineru_ready", "engine_status", "shutdown_engine"]

_VENV = Path(__file__).resolve().parents[3] / ".venv312" / "Scripts"
_LOCK = threading.Lock()
_ENGINE_PROC: subprocess.Popen | None = None   # 仅记录本模块拉起的实例
_LAST_USE = 0.0
_WATCHDOG_STARTED = False
_CONFIG_FIXED = False


def _exe(name: str) -> str:
    p = _VENV / name
    return str(p) if p.exists() else name


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run(cmd: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def engine_status() -> dict:
    """引擎状态（诊断用）：doclib / parse-server / 本模块管理的进程。"""
    port = int(os.environ.get("POLICYRAG_MINERU_PORT", "8889"))
    return {
        "doclib_running": _doclib_running(),
        "parse_server_port": port,
        "parse_server_up": _port_open("127.0.0.1", port),
        "managed_proc_alive": _ENGINE_PROC is not None and _ENGINE_PROC.poll() is None,
    }


def _doclib_running() -> bool:
    try:
        r = _run([_exe("mineru.exe"), "server", "status"], timeout=30)
        out = (r.stdout or "") + (r.stderr or "")
        # 注意："服务未运行"也含"运行"二字——必须先排除否定形态
        if "未运行" in out or "未启动" in out:
            return False
        return r.returncode == 0 and ("PID" in out)
    except Exception:  # noqa: BLE001
        return False


def _ensure_config() -> None:
    """配置自愈：本地 parse-server 模式与地址写入 mineru 配置（幂等）。"""
    global _CONFIG_FIXED
    if _CONFIG_FIXED:
        return
    try:
        _run([_exe("mineru.exe"), "config", "set", "parse_server.local.mode", "self_hosted"], 30)
        port = int(os.environ.get("POLICYRAG_MINERU_PORT", "8889"))
        _run([_exe("mineru.exe"), "config", "set", "parse_server.local.self_hosted_url",
              f"http://127.0.0.1:{port}"], 30)
        _CONFIG_FIXED = True
    except Exception:  # noqa: BLE001
        pass


def ensure_mineru_ready(timeout_s: float = 150) -> bool:
    """确保 MinerU 全栈就绪。返回 True=可用；False=不可用（调用方降级）。"""
    global _ENGINE_PROC, _LAST_USE, _WATCHDOG_STARTED
    if os.environ.get("POLICYRAG_MINERU_AUTOSTART", "1") == "0":
        return _port_open("127.0.0.1", int(os.environ.get("POLICYRAG_MINERU_PORT", "8889")))
    with _LOCK:
        port = int(os.environ.get("POLICYRAG_MINERU_PORT", "8889"))
        _ensure_config()
        # 1) doclib 守护（轻量）
        if not _doclib_running():
            try:
                _run([_exe("mineru.exe"), "server", "start"], 60)
            except Exception:  # noqa: BLE001
                return False
        # 2) parse-server 引擎（重）——已在跑则复用（不接管外部实例）
        if _port_open("127.0.0.1", port):
            _LAST_USE = time.time()
            _start_watchdog()
            return True
        env = dict(os.environ)
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("PYTHONUTF8", "1")
        try:
            flags = 0x00000008 if os.name == "nt" else 0  # DETACHED_PROCESS
            _ENGINE_PROC = subprocess.Popen(
                [_exe("mineru-api.exe"), "--port", str(port), "--tier", "basic",
                 "--allow-local-source", "--concurrency", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, creationflags=flags)
        except Exception:  # noqa: BLE001
            return False
        # 3) 就绪等待
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _port_open("127.0.0.1", port):
                _LAST_USE = time.time()
                _start_watchdog()
                return True
            if _ENGINE_PROC.poll() is not None:
                return False  # 进程退出 = 启动失败
            time.sleep(1.5)
        return False


def touch() -> None:
    global _LAST_USE
    _LAST_USE = time.time()


def _start_watchdog() -> None:
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    _WATCHDOG_STARTED = True
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()


def _watchdog_loop() -> None:
    """闲置释放：仅停掉本模块拉起的引擎，释放 ~1GB 内存。"""
    idle_s = float(os.environ.get("POLICYRAG_MINERU_IDLE_S", "1800"))
    while True:
        time.sleep(60)
        with _LOCK:
            if _ENGINE_PROC is None or _ENGINE_PROC.poll() is not None:
                continue
            if time.time() - _LAST_USE > idle_s:
                shutdown_engine()


def shutdown_engine() -> None:
    """停掉本模块拉起的 parse-server（外部启动的不动）。"""
    global _ENGINE_PROC
    with _LOCK:
        if _ENGINE_PROC is not None and _ENGINE_PROC.poll() is None:
            try:
                _ENGINE_PROC.terminate()
            except Exception:  # noqa: BLE001
                pass
        _ENGINE_PROC = None
