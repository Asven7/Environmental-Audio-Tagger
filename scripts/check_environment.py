#!/usr/bin/env python3
"""Report the local development environment without requiring project dependencies.

This script is intentionally standard-library only so it can be run before the
virtual environment is fully configured. If PyTorch/torchaudio are installed,
it also reports their accelerator status.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode, output


def _virtualenv_status() -> str:
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return "active" if active else "not active"


def _memory_gb() -> float | None:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024**3)
        except Exception:
            return None
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return (page_size * pages) / (1024**3)
        except (AttributeError, ValueError, OSError):
            return None
    return None


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed (version unavailable)"))


def _report_torch() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"torch: not importable ({exc})")
        return

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return

    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"CUDA device: {torch.cuda.get_device_name(index)}")
        print(f"CUDA device VRAM: {props.total_memory / (1024**3):.2f} GiB")
        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"CUDA smoke operation: PASS ({tuple(y.shape)})")
    except Exception as exc:
        print(f"CUDA smoke operation: FAIL ({exc})")


def main() -> int:
    print("=== Environmental Audio Tagger: Environment Check ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Virtual environment: {_virtualenv_status()}")
    print(f"CPU logical cores: {os.cpu_count()}")

    memory = _memory_gb()
    print(f"RAM: {memory:.2f} GiB" if memory is not None else "RAM: unavailable")

    project_root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(project_root)
    print(f"Project filesystem free space: {disk.free / (1024**3):.2f} GiB")

    git_code, git_output = _run(["git", "--version"])
    print(f"Git: {git_output if git_code == 0 else 'not available'}")

    nvidia_code, nvidia_output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )
    print(f"NVIDIA GPU: {nvidia_output if nvidia_code == 0 else 'not available'}")

    pip_code, pip_output = _run([sys.executable, "-m", "pip", "--version"])
    print(f"pip: {pip_output if pip_code == 0 else 'not available'}")

    print("\n--- Optional project dependencies ---")
    for package in ("numpy", "pandas", "scipy", "soundfile", "sklearn", "yaml", "matplotlib"):
        version = _package_version(package)
        print(f"{package}: {version or 'not installed'}")

    print(f"torchaudio: {_package_version('torchaudio') or 'not installed'}")
    _report_torch()

    print("\nEnvironment check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
